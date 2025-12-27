import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
import kornia.geometry as KG
import kornia.feature as KF

from utils.losses import TempConstLoss
from utils.torch_utils import conditional_no_grad
from utils.metrics import meanIoU, class_meanIoU, MetricMeter, PerClassMetricMeter
from utils.distributed import is_main_process
from data.utils.images_transforms import soft_to_hard_labels
from models.opt_flow import flowwarp, interpolate_flow
from models.video.mixing_layers import (
    STT3DMixingBlock,
    STT2DMixingBlock,
    Conv2DMixingBlock,
    Conv3DMixingBlock,
    IdMixingBlock)
from models.video.sim_layers import (
    ConstantSimBlock,
    AverageSimBlock,
    CossimSimBlock,
    ConvSimBlock,
    ConvSimBlockBase,
    ConvSimBlockv2,
    STT3DConvSimBlock,
    STT3DConvSimBlock2,
    STT3DCosSimBlock
)
from models.video.modules import Clip_PSP_Block, Clip_OCR_Block

def get_model(model_cfg, base_model, flow_model, n_classes):
    model_name = model_cfg.get("model_name", "base").lower()
    if model_name == "base":
        return ConsistencyWrapper(base_model, flow_model, n_classes, model_cfg)
    elif model_name == "feat":
        return ConsistencyFeatureWrapper(base_model, flow_model, n_classes, model_cfg)
    elif model_name == "dff":
        return DFFWrapper(base_model, flow_model, n_classes, model_cfg)
    elif model_name == "dff2":
        return DFF2Wrapper(base_model, flow_model, n_classes, model_cfg)
    elif model_name == "netwarp":
        return NetWarpWrapper(base_model, flow_model, n_classes, model_cfg)
    elif model_name == "tcb":
        return TCBWrapper(base_model, flow_model, n_classes, model_cfg)
    else:
        raise NotImplementedError()

def parse_mixing_ops(op_name, dim, n_layers, window_size=None, feat_mixing=False):
    op_name = op_name.lower()
    if op_name == "3dwmsa":
        if window_size is None:
            window_size = [2,7,7]
        return STT3DMixingBlock(dim, n_layers, window_size=window_size, shift_size=[0,3,3], n_heads=1, hidden_dim=4*dim)
    elif op_name == "2dwmsa":
        if window_size is None:
            window_size = [9,9]
        return STT2DMixingBlock(dim*2, n_layers, window_size=window_size, shift_size=[3,3], n_heads=1, hidden_dim=8*dim)
    elif op_name == "2d_conv":
        return Conv2DMixingBlock(dim*2, n_layers, hidden_dim=8*dim)
    elif op_name == "3d_conv":
        return Conv3DMixingBlock(dim, n_layers, hidden_dim=4*dim)
    elif op_name == "id":
        return IdMixingBlock()
    else:
        print(f"Mixing op {op_name} is not implemented")
        raise NotImplementedError()
    
def parse_sim_ops(op_name, embed_dim):
    op_name = op_name.lower()
    if op_name == "cossim":
        return CossimSimBlock()
    elif op_name == "conv":
        return ConvSimBlock(embed_dim)
    elif op_name == "convbase":
        return ConvSimBlockBase(embed_dim)
    elif op_name == "conv2":
        return ConvSimBlockv2(embed_dim)
    elif op_name == "constant":
        return ConstantSimBlock(0.)
    elif op_name == "avg":
        return AverageSimBlock(0.5)
    elif op_name == "3dwmsaconv":
        return STT3DConvSimBlock(embed_dim)
    elif op_name == "3dwmsaconv2":
        return STT3DConvSimBlock2(embed_dim)
    elif op_name == "3dwmsacos":
        return STT3DCosSimBlock(embed_dim)
    elif op_name == "reuse":
        return ConstantSimBlock(1.)
    else:
        print(f"Similarity op {op_name} is not implemented")
        raise NotImplementedError()


def get_matching_kpts(lafs1, lafs2, idxs):
    src_pts = KF.get_laf_center(lafs1).view(-1, 2)[idxs[:, 0]]
    dst_pts = KF.get_laf_center(lafs2).view(-1, 2)[idxs[:, 1]]
    return src_pts, dst_pts

class DiskFeatureRegistrator(nn.Module):
    def __init__(self, low_scale_homo):
        super().__init__()
        self.num_features = 2048
        self.disk = KF.DISK.from_pretrained("depth")
        self.lg_matcher = KF.LightGlueMatcher("disk").eval()
        self.ransac = KG.RANSAC(model_type="homography", inl_th=2.5)

        self.low_scale_homo = low_scale_homo

    def forward(self, last_frame, frame):
        if self.low_scale_homo:
            last_frame = nn.functional.interpolate(last_frame, scale_factor=1/4, mode="bilinear", antialias=True)
            frame = nn.functional.interpolate(frame, scale_factor=1/4, mode="bilinear", antialias=True)
        hw1 = torch.tensor(last_frame.shape[2:], device=frame.device)
        hw2 = torch.tensor(frame.shape[2:], device=frame.device)
        with torch.inference_mode():
            inp = torch.cat([last_frame, frame], dim=0)
            features1, features2 = self.disk(inp, self.num_features, pad_if_not_divisible=True)
            kps1, descs1 = features1.keypoints, features1.descriptors
            kps2, descs2 = features2.keypoints, features2.descriptors
            lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=frame.device))
            lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=frame.device))

            dists, idxs = self.lg_matcher(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)
        if len(idxs) >= 4:
            src_pts, dst_pts = get_matching_kpts(lafs1, lafs2, idxs)
            homo, mask = self.ransac(src_pts, dst_pts)
        else:
            homo = torch.tensor([[1.,0,0], [0,1.,0], [0,0,1.]], device=frame.device)
            print("not computed")
        return homo.unsqueeze(0)

    def batch_compute(self, last_frames, frames):
        homos = []
        for k in range(len(frames)):
            h = self(last_frames[k].unsqueeze(0), frames[k].unsqueeze(0)).detach()
            homos.append(h)
        return homos
    
class IdentityRegistrator(nn.Module):
    def __init__(self, low_scale_homo):
        super().__init__()
    def forward(self, last_frame, frame):
        homo = torch.tensor([[1.,0,0], [0,1.,0], [0,0,1.]], device=frame.device)
        return homo.unsqueeze(0)
    def batch_compute(self, last_frames, frames):
        homos = []
        for k in range(len(frames)):
            h = self(last_frames[k].unsqueeze(0), frames[k].unsqueeze(0)).detach()
            homos.append(h)
        return homos


def _normalize_homo(homo, device, dtype):
    if not torch.is_tensor(homo):
        homo = torch.as_tensor(homo, device=device, dtype=dtype)
    else:
        homo = homo.to(device=device, dtype=dtype)
    while homo.dim() > 2 and homo.shape[0] == 1:
        homo = homo.squeeze(0)
    if homo.dim() != 2 or homo.shape[-2:] != (3, 3):
        if homo.numel() == 9:
            homo = homo.reshape(3, 3)
            return homo, True
        return None, False
    return homo, True

def warp_batch(homos, last_out, scale=None, interp_mode="bilinear", det_eps=1e-8, fallback="identity", log_state=None, log_every=200):
    if homos is None:
        return last_out
    batch_size = last_out.shape[0]
    device = last_out.device
    dtype = last_out.dtype
    fallback = (fallback or "identity").lower()
    if torch.is_tensor(homos):
        dtype = homos.dtype
    elif isinstance(homos, (list, tuple)) and len(homos) > 0 and torch.is_tensor(homos[0]):
        dtype = homos[0].dtype

    if isinstance(homos, (list, tuple)):
        homo_list = list(homos)
        if len(homo_list) == 0:
            homo_list = [torch.eye(3, device=device, dtype=dtype) for _ in range(batch_size)]
        if len(homo_list) == 1 and batch_size > 1:
            homo_list = homo_list * batch_size
        if len(homo_list) < batch_size:
            homo_list = homo_list + [homo_list[-1]] * (batch_size - len(homo_list))
        homo_list = homo_list[:batch_size]
    elif torch.is_tensor(homos):
        if homos.dim() == 2:
            homo_list = [homos] * batch_size
        else:
            if homos.shape[0] == 1 and batch_size > 1:
                homo_list = [homos[0]] * batch_size
            else:
                homo_list = [homos[k] for k in range(batch_size)]
    else:
        homo_list = [torch.eye(3, device=device, dtype=dtype) for _ in range(batch_size)]

    if log_state is not None:
        log_state.setdefault("step", 0)
        log_state["step"] += 1
        log_state.setdefault("invalid", 0)
        log_state.setdefault("total", 0)
        log_state.setdefault("min_det", None)
        log_state.setdefault("max_det", None)

    warped_list = []
    identity = torch.eye(3, device=device, dtype=dtype)
    down = None
    up = None
    if scale is not None:
        down = torch.tensor([[[scale, 0, 0], [0, scale, 0], [0, 0, 1]]], device=device, dtype=dtype)
        up = torch.tensor([[[1 / scale, 0, 0], [0, 1 / scale, 0], [0, 0, 1]]], device=device, dtype=dtype)

    for k in range(batch_size):
        homo, shape_ok = _normalize_homo(homo_list[k], device, dtype)
        valid = False
        det_val = None
        if shape_ok and homo is not None:
            finite = torch.isfinite(homo).all().item()
            det = torch.linalg.det(homo.float())
            det_finite = torch.isfinite(det).item()
            if det_finite:
                det_val = det.item()
            valid = finite and det_finite and (abs(det_val) > det_eps if det_val is not None else False)

        if log_state is not None:
            log_state["total"] += 1
            if not valid:
                log_state["invalid"] += 1
            if det_val is not None:
                if log_state["min_det"] is None or det_val < log_state["min_det"]:
                    log_state["min_det"] = det_val
                if log_state["max_det"] is None or det_val > log_state["max_det"]:
                    log_state["max_det"] = det_val

        if not valid:
            if fallback == "skip":
                warped_list.append(last_out[k].unsqueeze(0))
                continue
            homo = identity

        if homo.dim() == 2:
            homo = homo.unsqueeze(0)
        if scale is not None:
            homo = down @ homo @ up
        warped_list.append(KG.warp_perspective(last_out[k].unsqueeze(0), homo, last_out[k].shape[-2:], mode=interp_mode))
    last_out_warped = torch.cat(warped_list, dim=0) 
    if log_state is not None and log_every and log_state["step"] % log_every == 0:
        if is_main_process():
            det_min = log_state["min_det"]
            det_max = log_state["max_det"]
            det_range = "n/a" if det_min is None else f"[{det_min:.3e}, {det_max:.3e}]"
            print(f"[warp_batch] invalid homos: {log_state['invalid']}/{log_state['total']} | det range: {det_range} | eps={det_eps} fallback={fallback}")
        log_state["invalid"] = 0
        log_state["total"] = 0
        log_state["min_det"] = None
        log_state["max_det"] = None
    return last_out_warped

def warp(homo, last_out, scale=None, interp_mode="bilinear", det_eps=1e-8, fallback="identity"):
    last_out_warped = warp_batch(homo, last_out, scale=scale, interp_mode=interp_mode, det_eps=det_eps, fallback=fallback)
    return last_out_warped


class ConsistencyWrapper(nn.Module):
    def __init__(
            self, 
            base_model,
            flow_model, 
            n_classes, 
            model_cfg,
            ):
        super().__init__()
        self.base_model = base_model
        self.flow_model = flow_model
        self.n_classes = n_classes
        self.pred_mixing_op = model_cfg["pred_mixing_op"]
        self.sim_op = model_cfg["sim_op"]
        self.n_mixing_layers = model_cfg.get("n_mixing_layers", 1)
        self.window_size = model_cfg.get("window_size", None)
        self.upsampling = model_cfg.get("upsampling", "bilinear")
        # Losses
        self.const_loss_lb = model_cfg.get("const_loss_lb", 0)
        self.const_loss = TempConstLoss()
        self.kd_consistent_labels = model_cfg.get("kd_consistent_labels", True)
        self.last_out_loss_lb = model_cfg.get("last_out_loss_lb", 0) # For KD only

        self.upsample_before_mixing = model_cfg.get("upsample_before_mixing", True)
        self.low_scale_homo = model_cfg.get("low_scale_homo", False)
        self.homo_det_eps = model_cfg.get("homo_det_eps", 1e-8)
        self.homo_fallback = model_cfg.get("homo_fallback", "identity")
        self.homo_log_every = model_cfg.get("homo_log_every", 200)
        self._homo_log_state = {"step": 0, "invalid": 0, "total": 0, "min_det": None, "max_det": None}

        self.features_interp_mode = model_cfg.get("features_interp_mode", "bilinear")
        self.k_registrator_model = model_cfg.get("k_registrator_model", None)
        if self.k_registrator_model == "disk":
            self.registrator = DiskFeatureRegistrator(self.low_scale_homo)
        elif self.k_registrator_model == "identity":
            self.registrator = IdentityRegistrator(self.low_scale_homo)
        else:
            self.registrator = None

        self.pred_mixing = parse_mixing_ops(self.pred_mixing_op, self.n_classes, self.n_mixing_layers, self.window_size)
        try:
            encoder_size = self.base_model.model.config.hidden_sizes[0]
        except:
            encoder_size = 96
        self.sim = parse_sim_ops(self.sim_op, encoder_size)


    def forward(self, frames, last_frames, homos=None, eval=False, return_last_out=False, return_sim=False):
        if homos is None:
            homos = self.registrator.batch_compute(last_frames, frames)

        with conditional_no_grad(eval):
            last_feats = self.base_model.encoder(last_frames)
            last_out = self.base_model.decoder(last_feats)
            feats = self.base_model.encoder(frames)
            out = self.base_model.decoder(feats)
            last_featmap, featmap = last_feats[0], feats[0]

            if self.upsample_before_mixing and out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            last_out_warped = warp_batch(
                homos,
                last_out,
                scale=None if (self.upsample_before_mixing or self.low_scale_homo) else 1/4,
                det_eps=self.homo_det_eps,
                fallback=self.homo_fallback,
                log_state=self._homo_log_state,
                log_every=self.homo_log_every,
            )
            last_featmap_warped = warp_batch(
                homos,
                last_featmap,
                scale=None if self.low_scale_homo else 1/4,
                interp_mode=self.features_interp_mode,
                det_eps=self.homo_det_eps,
                fallback=self.homo_fallback,
            )

            # Mixing
            last_out_warped, out = self.pred_mixing(last_out_warped, out)
            # Similarity map
            sim = self.sim(last_featmap_warped, featmap, size=out.shape[-2:])
            # Fusion based on similarity map
            out = sim*last_out_warped + (1-sim)*out

        if return_last_out:
            if return_sim:
                return out, last_out, sim
            return out, last_out
        return out
    
    def train_one_epoch(
            self,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
    ):
        train_loss = 0
        const_loss = 0
        const = None
        train_iter = tqdm(train_loader)
        self.train()
        for (frames, adj_frames, labels, homos) in train_iter:
            optimizer.zero_grad()
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)
            # Check if precomputed homos
            if homos.dim() < 3:
                homos = None
            else:
                homos = homos.to(device)

            last_frames = adj_frames[0]
            out, last_out = self.forward(frames, last_frames, homos=homos, eval=False, return_last_out=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                loss += const

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            train_iter.set_description(desc=f"train loss = {loss.item():.4f}, const = {const_value:.4f}")
        train_loss = train_loss/len(train_loader)
        const_loss = const_loss/len(train_loader)
        return train_loss, const_loss
    
    def evaluate_with_metrics(
            self,
            val_loader,
            criterion,
            device,
            n_classes,
            ignore_index=255
    ):
        val_loss = 0
        const_loss = 0
        const = None
        val_iter = tqdm(val_loader)
        mIoU1 = MetricMeter()
        mIoU2 = MetricMeter()
        #classes_mIoU = PerClassMetricMeter(n_classes)
        preds_for_iou = []
        labels_for_iou = []
        self.eval()
        for (frames, adj_frames, labels, homos) in val_iter:
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)

            # Check if precomputed homos
            if homos.dim() < 3:
                homos = None
            else:
                homos = homos.to(device)
            
            last_frames = adj_frames[0]
            out, last_out = self.forward(frames, last_frames, homos=homos, eval=True, return_last_out=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                loss += const

            preds = out.argmax(1)
            preds = preds.detach().cpu()
            if labels.dim() != preds.dim():
                labels = soft_to_hard_labels(labels, ignore_index)
            labels = labels.detach().cpu()
            preds_for_iou.append(preds)
            labels_for_iou.append(labels)

            val_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            val_iter.set_description(desc=f"val loss = {loss.item():.4f}, const = {const_value:.4f}")

        preds_for_iou = torch.cat(preds_for_iou, dim=0)
        labels_for_iou = torch.cat(labels_for_iou, dim=0)
        global_miou = meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        val_loss = val_loss/len(val_loader)
        const_loss = const_loss/len(val_loader)

        results = {
            "val_loss": val_loss,
            "const_loss": const_loss,
            "mIoU1": mIoU1.avg,
            "mIoU2": mIoU2.avg,
            "global_miou": global_miou,
            "classes_mIoU": global_classes_iou
        }

        return results
    
    def kd_train_one_epoch(
            self,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
    ):
        train_loss = 0
        const_loss = 0
        const = None
        train_iter = tqdm(train_loader)
        self.train()
        for (frames, adj_frames, labels, adj_labels, homos) in train_iter:
            optimizer.zero_grad()
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)
            adj_labels = [adj_l.to(device) for adj_l in adj_labels]
            # Check if precomputed homos
            if homos.dim() < 3:
                homos = None
            else:
                homos = homos.to(device)

            last_frames = adj_frames[0]
            last_labels = adj_labels[0]
            out, last_out, sim = self.forward(frames, last_frames, homos=homos, eval=False, return_last_out=True, return_sim=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                sim = nn.functional.interpolate(sim, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            if self.kd_consistent_labels:
                last_labels, labels = self.kd_get_consistent_logits(last_frames, frames, last_labels, labels)

            loss = criterion(out, labels) + self.last_out_loss_lb * criterion(last_out, last_labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                loss += const

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            train_iter.set_description(desc=f"train loss = {loss.item():.4f}, const = {const_value:.4f}")
        train_loss = train_loss/len(train_loader)
        const_loss = const_loss/len(train_loader)
        return train_loss, const_loss
    
    def kd_get_consistent_logits(self, last_frames, frames, last_labels, labels):
        with torch.no_grad():
            self.flow_model.eval()
            _, flow_last2current = self.flow_model(frames, last_frames, iters=15, test_mode=True)
            _, flow_current2last = self.flow_model(last_frames, frames, iters=15, test_mode=True)
        occlusion_mask = (((flow_last2current + flow_current2last)**2).sum(1) > 0.01*((flow_last2current**2).sum(1) + (flow_current2last**2).sum(1)) + 0.5).unsqueeze(1)

        warped_last_labels = flowwarp(last_labels, flow_last2current)
        warped_labels = flowwarp(labels, flow_current2last)

        last_labels_consit = (last_labels + warped_labels*(~occlusion_mask)) / (2*(~occlusion_mask) + 1*occlusion_mask)
        labels_consit = (labels + warped_last_labels*(~occlusion_mask)) / (2*(~occlusion_mask) + 1*occlusion_mask)
        return last_labels_consit, labels_consit
    
    def infer_frame(self, frame, homo=None, last_frame=None, last_out=None, last_feats=None):
        with torch.no_grad():
            feats = self.base_model.encoder(frame)
            out = self.base_model.decoder(feats)
        
        if last_out is not None:
            last_featmap, featmap = last_feats[0], feats[0]
            if homo is None:
                homo = self.registrator(last_frame, frame)
            
            with torch.no_grad():
                if self.upsample_before_mixing and out.shape[-2:] != frame.shape[-2:]:
                    out = nn.functional.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)
                    last_out = nn.functional.interpolate(last_out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)

                last_out_warped = warp(
                    homo,
                    last_out,
                    scale=None if (self.upsample_before_mixing or self.low_scale_homo) else 1/4,
                    det_eps=self.homo_det_eps,
                    fallback=self.homo_fallback,
                )
                last_featmap_warped = warp(
                    homo,
                    last_featmap,
                    scale=None if self.low_scale_homo else 1/4,
                    interp_mode=self.features_interp_mode,
                    det_eps=self.homo_det_eps,
                    fallback=self.homo_fallback,
                )

                last_out_warped, out = self.pred_mixing(last_out_warped, out)
                sim = self.sim(last_featmap_warped, featmap, size=out.shape[-2:])
                out = sim*last_out_warped + (1-sim)*out

        return out, feats
    
    def infer_video(self, frames, homos, device):
        self.eval()
        preds = []
        last_frame = None
        last_out = None
        last_feats = None
        for (i, frame) in enumerate(frames):
            frame = frame.unsqueeze(0).to(device)
            homo = homos[i]
            if homo is not None:
                homo = homo.to(device)
            out, feats = self.infer_frame(frame, homo, last_frame, last_out, last_feats)
            last_feats = feats
            last_out = out
            last_frame = frame

            if out.shape[-2:] != frame.shape[-2:]:
                out = nn.functional.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)
            pred = out.argmax(1).squeeze().detach().cpu()
            preds.append(pred)
        return preds
    

class ConsistencyFeatureWrapper(ConsistencyWrapper):
    def __init__(
            self, 
            base_model,
            flow_model, 
            n_classes, 
            model_cfg,
            ):
        super().__init__(base_model, flow_model, n_classes, model_cfg)
        embed_dim = 512
        self.pred_mixing = parse_mixing_ops(self.pred_mixing_op, embed_dim, self.n_mixing_layers, self.window_size, feat_mixing=True)


    def forward(self, frames, last_frames, homos=None, eval=False, return_last_out=False, return_sim=False):
        if homos is None:
            homos = self.registrator.batch_compute(last_frames, frames)

        with conditional_no_grad(eval):
            last_feats = self.base_model.encoder(last_frames)
            last_decoderfeats = self.base_model.decoder_lastfeat(last_feats)
            feats = self.base_model.encoder(frames)
            decoderfeats = self.base_model.decoder_lastfeat(feats)
            last_featmap, featmap = last_feats[0], feats[0]

            last_decoderfeats_warped = warp_batch(
                homos,
                last_decoderfeats,
                scale=None if self.low_scale_homo else 1/4,
                interp_mode=self.features_interp_mode,
                det_eps=self.homo_det_eps,
                fallback=self.homo_fallback,
                log_state=self._homo_log_state,
                log_every=self.homo_log_every,
            )
            last_featmap_warped = warp_batch(
                homos,
                last_featmap,
                scale=None if self.low_scale_homo else 1/4,
                interp_mode=self.features_interp_mode,
                det_eps=self.homo_det_eps,
                fallback=self.homo_fallback,
            )

            # Mixing
            last_decoderfeats_warped, decoderfeats = self.pred_mixing(last_decoderfeats_warped, decoderfeats)
            # Similarity map
            sim = self.sim(last_featmap_warped, featmap, size=decoderfeats.shape[-2:])
            # Fusion based on similarity map
            #decoderfeats = sim*last_decoderfeats_warped + (1-sim)*decoderfeats
            # Final classifying layer
            out = self.base_model.forward_classifier(decoderfeats)
            last_out_warped = self.base_model.forward_classifier(last_decoderfeats_warped)
            # Fusion on predictions
            out = sim*last_out_warped + (1-sim)*out

        if return_last_out:
            last_out = self.base_model.forward_classifier(last_decoderfeats)
            if return_sim:
                return out, last_out, sim
            return out, last_out
        return out
    
    
    def infer_frame(self, frame, homo=None, last_frame=None, last_decoderfeats=None, last_feats=None):
        with torch.no_grad():
            feats = self.base_model.encoder(frame)
            decoderfeats = self.base_model.decoder_lastfeat(feats)
        
        if last_decoderfeats is not None:
            last_featmap, featmap = last_feats[0], feats[0]
            if homo is None:
                homo = self.registrator(last_frame, frame)
            
            with torch.no_grad():
                last_decoderfeats_warped = warp(
                    homo,
                    last_decoderfeats,
                    scale=None if self.low_scale_homo else 1/4,
                    interp_mode=self.features_interp_mode,
                    det_eps=self.homo_det_eps,
                    fallback=self.homo_fallback,
                )
                last_featmap_warped = warp(
                    homo,
                    last_featmap,
                    scale=None if self.low_scale_homo else 1/4,
                    interp_mode=self.features_interp_mode,
                    det_eps=self.homo_det_eps,
                    fallback=self.homo_fallback,
                )

                last_decoderfeats_warped, decoderfeats = self.pred_mixing(last_decoderfeats_warped, decoderfeats)
                sim = self.sim(last_featmap_warped, featmap, size=decoderfeats.shape[-2:])
                #decoderfeats = sim*last_decoderfeats_warped + (1-sim)*decoderfeats
                out = self.base_model.forward_classifier(decoderfeats)
                last_out_warped = self.base_model.forward_classifier(last_decoderfeats_warped)
                out = sim*last_out_warped + (1-sim)*out
        else:
            out = self.base_model.forward_classifier(decoderfeats)

        return out, feats, decoderfeats
    
    def infer_video(self, frames, homos, K, device):
        self.eval()
        preds = []
        last_frame = None
        last_decoderfeats = None
        last_feats = None
        for (i, frame) in enumerate(frames):
            frame = frame.unsqueeze(0).to(device)
            homo = homos[i]
            if homo is not None:
                homo = homo.to(device)
            out, feats, decoderfeats = self.infer_frame(frame, homo, last_frame, last_decoderfeats, last_feats)
            last_feats = feats
            last_decoderfeats = decoderfeats
            last_frame = frame

            if out.shape[-2:] != frame.shape[-2:]:
                out = nn.functional.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)
            pred = out.argmax(1).squeeze().detach().cpu()
            preds.append(pred)
        return preds
    

class DFFWrapper(nn.Module):
    def __init__(
            self, 
            base_model,
            flow_model, 
            n_classes, 
            model_cfg,
            ):
        super().__init__()
        self.base_model = base_model
        self.flow_model = flow_model
        self.n_classes = n_classes
        self.upsampling = model_cfg.get("upsampling", "bilinear")
        # Losses
        self.const_loss_lb = model_cfg.get("const_loss_lb", 0)
        self.const_loss = TempConstLoss()
        self.train_flow = model_cfg.get("train_flow", True)
        self.n_iters = model_cfg.get("flow_iters", 10)
        self.k_interval = model_cfg.get("k_interval", 4)
        self.flow_mode = model_cfg.get("flow_mode", "bilinear")


    def forward(self, frames, last_frames, eval=False, return_last_out=False):
        if self.train_flow:
            self.flow_model.train()
            _, flow = self.flow_model(frames, last_frames, iters=self.n_iters, test_mode=True)
        else:
            with torch.no_grad():
                self.flow_model.eval()
                _, flow = self.flow_model(frames, last_frames, iters=self.n_iters, test_mode=True)
                flow = flow.detach()

        with conditional_no_grad(eval):
            last_feats = self.base_model.encoder(last_frames)
            last_out = self.base_model.decoder(last_feats)
            last_feats_warped = [flowwarp(f, interpolate_flow(flow, f.shape[-1]/flow.shape[-1]), self.flow_mode) for f in last_feats]
            out = self.base_model.decoder(last_feats_warped)

        if return_last_out:
            return out, last_out
        return out
    
    def train_one_epoch(
            self,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
    ):
        train_loss = 0
        const_loss = 0
        const = None
        train_iter = tqdm(train_loader)
        self.train()
        for (frames, adj_frames, labels, homos) in train_iter:
            optimizer.zero_grad()
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)

            last_frames = adj_frames[0]
            out, last_out = self.forward(frames, last_frames, eval=False, return_last_out=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                loss += const

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            train_iter.set_description(desc=f"train loss = {loss.item():.4f}, const = {const_value:.4f}")
        train_loss = train_loss/len(train_loader)
        const_loss = const_loss/len(train_loader)
        return train_loss, const_loss
    
    def evaluate_with_metrics(
            self,
            val_loader,
            criterion,
            device,
            n_classes,
            ignore_index=255
    ):
        val_loss = 0
        const_loss = 0
        const = None
        val_iter = tqdm(val_loader)
        mIoU1 = MetricMeter()
        mIoU2 = MetricMeter()
        #classes_mIoU = PerClassMetricMeter(n_classes)
        preds_for_iou = []
        labels_for_iou = []
        self.eval()
        for (frames, adj_frames, labels, homos) in val_iter:
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)
            
            last_frames = adj_frames[0]
            out, last_out = self.forward(frames, last_frames, eval=True, return_last_out=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                loss += const

            preds = out.argmax(1)
            preds = preds.detach().cpu()
            if labels.dim() != preds.dim():
                labels = soft_to_hard_labels(labels, ignore_index)
            labels = labels.detach().cpu()
            preds_for_iou.append(preds)
            labels_for_iou.append(labels)

            val_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            val_iter.set_description(desc=f"val loss = {loss.item():.4f}, const = {const_value:.4f}")

        preds_for_iou = torch.cat(preds_for_iou, dim=0)
        labels_for_iou = torch.cat(labels_for_iou, dim=0)
        global_miou = meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        val_loss = val_loss/len(val_loader)
        const_loss = const_loss/len(val_loader)

        results = {
            "val_loss": val_loss,
            "const_loss": const_loss,
            "mIoU1": mIoU1.avg,
            "mIoU2": mIoU2.avg,
            "global_miou": global_miou,
            "classes_mIoU": global_classes_iou
        }

        return results
    
    
    def infer_frame(self, frame, last_frame=None, last_feats=None):
        if last_feats is not None:
            with torch.no_grad():
                self.flow_model.eval()
                _, flow = self.flow_model(frame, last_frame, iters=self.n_iters, test_mode=True)
                flow = flow.detach()
            
            with torch.no_grad():
                last_feats_warped = [flowwarp(f, interpolate_flow(flow, f.shape[-1]/flow.shape[-1]), self.flow_mode) for f in last_feats]
                out = self.base_model.decoder(last_feats_warped)
            
            return out, last_feats_warped

        else:
            with torch.no_grad():
                feats = self.base_model.encoder(frame)
                out = self.base_model.decoder(feats)

            return out, feats
    
    def infer_video(self, frames, homos, device):
        self.eval()
        preds = []
        last_frame = None
        last_feats = None
        for (i, frame) in enumerate(frames):
            frame = frame.unsqueeze(0).to(device)
            if i%self.k_interval == 0:
                out, feats = self.infer_frame(frame, last_frame, last_feats=None)
            else:
                out, feats = self.infer_frame(frame, last_frame, last_feats=last_feats)
            last_feats = feats
            last_frame = frame

            if out.shape[-2:] != frame.shape[-2:]:
                out = nn.functional.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)
            pred = out.argmax(1).squeeze().detach().cpu()
            preds.append(pred)
        return preds
    

class DFF2Wrapper(DFFWrapper):
    def __init__(
            self, 
            base_model,
            flow_model, 
            n_classes, 
            model_cfg,
            ):
        super().__init__(base_model, flow_model, n_classes, model_cfg)
    
    def forward(self, frames, last_frames, eval=False, return_last_out=False):
        if self.train_flow:
            self.flow_model.train()
            _, flow = self.flow_model(frames, last_frames, iters=self.n_iters, test_mode=True)
        else:
            with torch.no_grad():
                self.flow_model.eval()
                _, flow = self.flow_model(frames, last_frames, iters=self.n_iters, test_mode=True)
                flow = flow.detach()

        with conditional_no_grad(eval):
            last_feats = self.base_model.encoder(last_frames)
            last_decoderfeats = self.base_model.decoder_lastfeat(last_feats)
            last_out = self.base_model.forward_classifier(last_decoderfeats)
            last_decoderfeats_warped = flowwarp(last_decoderfeats, interpolate_flow(flow, 1/4), self.flow_mode)
            out = self.base_model.forward_classifier(last_decoderfeats_warped)

        if return_last_out:
            return out, last_out
        return out
    
    def infer_frame(self, frame, last_frame=None, last_decoderfeats=None):
        if last_decoderfeats is not None:
            with torch.no_grad():
                self.flow_model.eval()
                _, flow = self.flow_model(frame, last_frame, iters=self.n_iters, test_mode=True)
                flow = flow.detach()
            
            with torch.no_grad():
                last_decoderfeats_warped = flowwarp(last_decoderfeats, interpolate_flow(flow, 1/4), self.flow_mode)
                out = self.base_model.forward_classifier(last_decoderfeats_warped)

            return out, last_decoderfeats_warped

        else:
            with torch.no_grad():
                feats = self.base_model.encoder(frame)
                decoderfeats = self.base_model.decoder_lastfeat(feats)
                out = self.base_model.forward_classifier(decoderfeats)

            return out, decoderfeats
    
    
    def infer_video(self, frames, homos, device):
        self.eval()
        preds = []
        last_frame = None
        last_decoderfeats = None
        for (i, frame) in enumerate(frames):
            frame = frame.unsqueeze(0).to(device)
            if i%self.k_interval == 0:
                out, decoderfeats = self.infer_frame(frame, last_frame, last_decoderfeats=None)
            else:
                out, decoderfeats = self.infer_frame(frame, last_frame, last_decoderfeats=last_decoderfeats)
            last_decoderfeats = decoderfeats
            last_frame = frame

            if out.shape[-2:] != frame.shape[-2:]:
                out = nn.functional.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)
            pred = out.argmax(1).squeeze().detach().cpu()
            preds.append(pred)
        return preds


class FlowCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(11, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(4, 2, 3, padding=1),
        )


    def forward(self, flow, frame, last_frame):
        x = torch.cat([flow, last_frame, frame, last_frame-frame], dim=1)
        x = self.conv_1(x)
        x = torch.cat([x, flow], dim=1)
        x = self.conv_2(x)
        return x


class NetWarpWrapper(nn.Module):
    def __init__(
            self, 
            base_model,
            flow_model, 
            n_classes, 
            model_cfg,
            ):
        super().__init__()
        self.base_model = base_model
        self.flow_model = flow_model
        self.n_classes = n_classes
        self.upsampling = model_cfg.get("upsampling", "bilinear")
        # Losses
        self.const_loss_lb = model_cfg.get("const_loss_lb", 0)
        self.const_loss = TempConstLoss()
        self.train_flow = model_cfg.get("train_flow", True)
        self.n_iters = model_cfg.get("flow_iters", 10)
        self.flow_mode = model_cfg.get("flow_mode", "bilinear")
        self.flow_cnn = FlowCNN()
        self.past_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 96, 1, 1)),
            nn.Parameter(torch.zeros(1, 192, 1, 1)),
            nn.Parameter(torch.zeros(1, 384, 1, 1)),
            nn.Parameter(torch.zeros(1, 768, 1, 1))
        ])
        self.current_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1, 96, 1, 1)),
            nn.Parameter(torch.ones(1, 192, 1, 1)),
            nn.Parameter(torch.ones(1, 384, 1, 1)),
            nn.Parameter(torch.ones(1, 768, 1, 1))
        ])

    def forward(self, frames, last_frames, eval=False, return_last_out=False):
        if self.train_flow:
            self.flow_model.train()
            _, flow = self.flow_model(frames, last_frames, iters=self.n_iters, test_mode=True)
        else:
            with torch.no_grad():
                self.flow_model.eval()
                _, flow = self.flow_model(frames, last_frames, iters=self.n_iters, test_mode=True)
                flow = flow.detach()

        with conditional_no_grad(eval):
            flow = self.flow_cnn(flow, frames, last_frames)

            last_feats = self.base_model.encoder(last_frames)
            last_out = self.base_model.decoder(last_feats)
            feats = self.base_model.encoder(frames)

            last_feats_warped = [flowwarp(f, interpolate_flow(flow, f.shape[-1]/flow.shape[-1]), self.flow_mode) for f in last_feats]

            combined_feats = []
            for k in range(len(feats)):
                combined_feats.append(self.current_weights[k] * feats[k] + self.past_weights[k] * last_feats_warped[k])

            out = self.base_model.decoder(combined_feats)

        if return_last_out:
            return out, last_out
        return out
    
    def train_one_epoch(
            self,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
    ):
        train_loss = 0
        const_loss = 0
        const = None
        train_iter = tqdm(train_loader)
        self.train()
        for (frames, adj_frames, labels, homos) in train_iter:
            optimizer.zero_grad()
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)

            last_frames = adj_frames[0]
            out, last_out = self.forward(frames, last_frames, eval=False, return_last_out=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                loss += const

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            train_iter.set_description(desc=f"train loss = {loss.item():.4f}, const = {const_value:.4f}")
        train_loss = train_loss/len(train_loader)
        const_loss = const_loss/len(train_loader)
        return train_loss, const_loss
    
    def evaluate_with_metrics(
            self,
            val_loader,
            criterion,
            device,
            n_classes,
            ignore_index=255
    ):
        val_loss = 0
        const_loss = 0
        const = None
        val_iter = tqdm(val_loader)
        mIoU1 = MetricMeter()
        mIoU2 = MetricMeter()
        #classes_mIoU = PerClassMetricMeter(n_classes)
        preds_for_iou = []
        labels_for_iou = []
        self.eval()
        for (frames, adj_frames, labels, homos) in val_iter:
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)
            
            last_frames = adj_frames[0]
            out, last_out = self.forward(frames, last_frames, eval=True, return_last_out=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                loss += const

            preds = out.argmax(1)
            preds = preds.detach().cpu()
            if labels.dim() != preds.dim():
                labels = soft_to_hard_labels(labels, ignore_index)
            labels = labels.detach().cpu()
            preds_for_iou.append(preds)
            labels_for_iou.append(labels)

            val_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            val_iter.set_description(desc=f"val loss = {loss.item():.4f}, const = {const_value:.4f}")

        preds_for_iou = torch.cat(preds_for_iou, dim=0)
        labels_for_iou = torch.cat(labels_for_iou, dim=0)
        global_miou = meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        val_loss = val_loss/len(val_loader)
        const_loss = const_loss/len(val_loader)

        results = {
            "val_loss": val_loss,
            "const_loss": const_loss,
            "mIoU1": mIoU1.avg,
            "mIoU2": mIoU2.avg,
            "global_miou": global_miou,
            "classes_mIoU": global_classes_iou
        }

        return results
    
    
    def infer_frame(self, frame, last_frame=None, last_feats=None):
        if last_feats is not None:
            with torch.no_grad():
                self.flow_model.eval()
                _, flow = self.flow_model(frame, last_frame, iters=self.n_iters, test_mode=True)
                flow = flow.detach()
            
            with torch.no_grad():
                flow = self.flow_cnn(flow, frame, last_frame)

                feats = self.base_model.encoder(frame)
                last_feats_warped = [flowwarp(f, interpolate_flow(flow, f.shape[-1]/flow.shape[-1]), self.flow_mode) for f in last_feats]

                combined_feats = []
                for k in range(len(feats)):
                    combined_feats.append(self.current_weights[k] * feats[k] + self.past_weights[k] * last_feats_warped[k])

                out = self.base_model.decoder(combined_feats)
            
            return out, combined_feats

        else:
            with torch.no_grad():
                feats = self.base_model.encoder(frame)
                out = self.base_model.decoder(feats)

            return out, feats
    
    def infer_video(self, frames, homos, device):
        self.eval()
        preds = []
        last_frame = None
        last_feats = None
        for (i, frame) in enumerate(frames):
            frame = frame.unsqueeze(0).to(device)
            out, feats = self.infer_frame(frame, last_frame, last_feats=last_feats)
            last_feats = feats
            last_frame = frame

            if out.shape[-2:] != frame.shape[-2:]:
                out = nn.functional.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)
            pred = out.argmax(1).squeeze().detach().cpu()
            preds.append(pred)
        return preds
    

class TCBWrapper(nn.Module):
    def __init__(
            self, 
            base_model,
            flow_model, 
            n_classes, 
            model_cfg,
            ):
        super().__init__()
        self.base_model = base_model
        self.flow_model = flow_model
        self.n_classes = n_classes
        self.upsampling = model_cfg.get("upsampling", "bilinear")
        # Losses
        self.const_loss_lb = model_cfg.get("const_loss_lb", 0)
        self.const_loss = TempConstLoss()
        self.tcb = Clip_PSP_Block(768, 768) if model_cfg.get("tcb_module", "ocr") == "psp" else Clip_OCR_Block(768, 768, self.n_classes, use_memory=False)
        self.tcb_conv = nn.Conv2d(768*2,768,1)

    def forward(self, frames, adj_frames, eval=False):
        with conditional_no_grad(eval):
            B = frames.size(0)
            all_frames = torch.cat([frames] + adj_frames, dim=0)
            all_feats = self.base_model.encoder(all_frames)
            feats = [all_f[:B] for all_f in all_feats]
            adj_feats = [all_f[B:].split(B, dim=0) for all_f in all_feats] # backprop on the adj frames

            featmap = feats[-1]
            adj_featmaps = list(adj_feats[-1])
            featmap_tcb = self.tcb(featmap, adj_featmaps)
            featmap = self.tcb_conv(torch.cat([featmap, featmap_tcb], dim=1))
            feats[-1] = featmap

            out = self.base_model.decoder(feats)

        return out
    
    def train_one_epoch(
            self,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
    ):
        train_loss = 0
        const_loss = 0
        const = None
        train_iter = tqdm(train_loader)
        self.train()
        for (frames, adj_frames, labels, homos) in train_iter:
            optimizer.zero_grad()
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)

            out = self.forward(frames, adj_frames, eval=False)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            train_iter.set_description(desc=f"train loss = {loss.item():.4f}, const = {const_value:.4f}")
        train_loss = train_loss/len(train_loader)
        const_loss = const_loss/len(train_loader)
        return train_loss, const_loss
    
    def evaluate_with_metrics(
            self,
            val_loader,
            criterion,
            device,
            n_classes,
            ignore_index=255
    ):
        val_loss = 0
        const_loss = 0
        const = None
        val_iter = tqdm(val_loader)
        mIoU1 = MetricMeter()
        mIoU2 = MetricMeter()
        #classes_mIoU = PerClassMetricMeter(n_classes)
        preds_for_iou = []
        labels_for_iou = []
        self.eval()
        for (frames, adj_frames, labels, homos) in val_iter:
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)
            
            out = self.forward(frames, adj_frames, eval=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)

            preds = out.argmax(1)
            preds = preds.detach().cpu()
            if labels.dim() != preds.dim():
                labels = soft_to_hard_labels(labels, ignore_index)
            labels = labels.detach().cpu()
            preds_for_iou.append(preds)
            labels_for_iou.append(labels)

            val_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            val_iter.set_description(desc=f"val loss = {loss.item():.4f}, const = {const_value:.4f}")

        preds_for_iou = torch.cat(preds_for_iou, dim=0)
        labels_for_iou = torch.cat(labels_for_iou, dim=0)
        global_miou = meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        val_loss = val_loss/len(val_loader)
        const_loss = const_loss/len(val_loader)

        results = {
            "val_loss": val_loss,
            "const_loss": const_loss,
            "mIoU1": mIoU1.avg,
            "mIoU2": mIoU2.avg,
            "global_miou": global_miou,
            "classes_mIoU": global_classes_iou
        }

        return results
    
    
    def infer_frame(self, frame, adj_frames=None):
        if adj_frames is not None:
            with torch.no_grad():
                B = frame.size(0)
                all_frames = torch.cat([frame] + adj_frames, dim=0)
                all_feats = self.base_model.encoder(all_frames)
                feats = [all_f[:B] for all_f in all_feats]
                adj_feats = [all_f[B:].detach().split(B, dim=0) for all_f in all_feats]

                featmap = feats[-1]
                adj_featmaps = list(adj_feats[-1])
                featmap_tcb = self.tcb(featmap, adj_featmaps)
                featmap = self.tcb_conv(torch.cat([featmap, featmap_tcb], dim=1))
                feats[-1] = featmap

                out = self.base_model.decoder(feats)
            
            return out

        else:
            with torch.no_grad():
                out = self.base_model(frame)

            return out
    
    def infer_video(self, frames, homos, device):
        self.eval()
        preds = []
        for (i, frame) in enumerate(frames):
            frame = frame.unsqueeze(0).to(device)
            if i>2:
                adj_frames = [frames[i-j].unsqueeze(0).to(device) for j in range(1,4)]
                out = self.infer_frame(frame, adj_frames=adj_frames)
            else:
                out = self.infer_frame(frame, adj_frames=None)

            if out.shape[-2:] != frame.shape[-2:]:
                out = nn.functional.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)
            pred = out.argmax(1).squeeze().detach().cpu()
            preds.append(pred)
        return preds
