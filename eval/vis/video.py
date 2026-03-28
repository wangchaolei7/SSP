import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import os
    import PIL.Image as Image
    import numpy as np
    import time
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import yaml
    import time

    from vis_utils.visualization import color_predictions, inverse_normalize, pred_to_mask
    from models.image.models import get_model as get_image_model
    from models.opt_flow import get_flow_model
    from models.video.models_consistency import get_model as get_video_model
    from data.dataset_prep import prep_infer_image_dataset
    from data.datasets import APOLLOSCAPE, KITTI360, CAMVID
    from data.folder_sequence_dataset import folder_sequence_defaults, is_folder_sequence_dataset
    from utils.distributed import (
        setup_distributed,
        cleanup_distributed,
        is_distributed,
        is_main_process,
        get_rank,
        get_world_size,
        get_local_rank,
        barrier,
        all_reduce_tensor,
        merge_rank_outputs,
        SequenceDistributedSampler,
    )


def _update_confusion(confusion, pred, label, n_classes, ignore_index):
    pred = torch.as_tensor(pred).view(-1)
    label = torch.as_tensor(label).view(-1)
    if ignore_index is not None:
        mask = label != ignore_index
        pred = pred[mask]
        label = label[mask]
    if pred.numel() == 0:
        return confusion
    idx = label * n_classes + pred
    confusion += torch.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    return confusion

MVC_STRIDE = 4
MVC_WINDOWS = (8, 16)
CITYS_SIM_THRESH = 20.0


def _dataset_defaults(name):
    name = (name or "").lower()
    mapping = {
        "apolloscape": APOLLOSCAPE,
        "kitti360": KITTI360,
        "camvid": CAMVID,
    }
    ds = mapping.get(name)
    if ds is not None:
        return {
            "path": ds.path,
            "frame_folder": ds.frame_folder,
            "mask_folder": ds.mask_folder,
            "label_suffix": getattr(ds, "label_suffix", ""),
            "img_extension": ds.img_extension,
            "label_extension": ds.label_extension,
            "num_classes": ds.n_classes,
            "ignore_index": ds.ignore_index,
        }
    if is_folder_sequence_dataset(name):
        return folder_sequence_defaults(name)
    return {}


def _resolve_image_config_path(image_save_dir, img_checkpoint_folder, img_checkpoint_name, config_dir):
    ckpt_name = img_checkpoint_name.split("@")[-1]
    primary = os.path.join(
        image_save_dir,
        img_checkpoint_folder + img_checkpoint_name,
        f"{ckpt_name}_config.yaml",
    )
    if os.path.isfile(primary):
        return primary

    # Try to locate under the source-domain root (e.g., .../apollo/image/<ckpt>/...)
    parent = os.path.dirname(config_dir)
    source_root = os.path.dirname(parent) if os.path.basename(parent) in ("video", "image") else None
    if source_root:
        candidate = os.path.join(source_root, "image", img_checkpoint_name, f"{ckpt_name}_config.yaml")
        if os.path.isfile(candidate):
            return candidate
        # Fallback: any image config under source_root/image/*
        image_root = os.path.join(source_root, "image")
        if os.path.isdir(image_root):
            configs = []
            for entry in os.listdir(image_root):
                cfg_path = os.path.join(image_root, entry, f"{entry}_config.yaml")
                if os.path.isfile(cfg_path):
                    configs.append(cfg_path)
            if configs:
                configs.sort()
                return configs[-1]

    return None


def _to_gray_small(frame_chw, h_small, w_small):
    img = inverse_normalize(frame_chw)
    img_gray = Image.fromarray(img).convert("L")
    img_gray = img_gray.resize((w_small, h_small), Image.BILINEAR)
    return np.array(img_gray, dtype=np.uint8)


def _compute_vc_dense(preds, gts, n, ignore_index):
    L = len(preds)
    if L < n:
        return None
    vals = []
    for start in range(0, L - n + 1):
        pred_win = preds[start : start + n]
        gt_win = gts[start : start + n]
        gt0 = gt_win[0]
        gt_equal = np.ones(gt0.shape, dtype=bool)
        gt_valid = np.ones(gt0.shape, dtype=bool)
        for gt in gt_win:
            gt_valid &= (gt != ignore_index)
        for gt in gt_win[1:]:
            gt_equal &= (gt == gt0)
        gt_common = gt_equal & gt_valid
        denom = int(gt_common.sum())
        if denom == 0:
            continue
        pred0 = pred_win[0]
        pred_equal = np.ones(pred0.shape, dtype=bool)
        for p in pred_win[1:]:
            pred_equal &= (p == pred0)
        num = int((gt_common & pred_equal & (pred0 == gt0)).sum())
        vals.append(num / denom)
    if not vals:
        return None
    return float(np.mean(vals))


def _compute_vc_sparse_by_valid_windows(preds, gts, n, ignore_index):
    L = len(preds)
    if L < n:
        return None
    vals = []
    for start in range(0, L - n + 1):
        gt_win = gts[start : start + n]
        if any(x is None for x in gt_win):
            continue
        vc = _compute_vc_dense(
            preds[start : start + n],
            [x for x in gt_win if x is not None],
            n=n,
            ignore_index=ignore_index,
        )
        if vc is not None:
            vals.append(vc)
    if not vals:
        return None
    return float(np.mean(vals))


def _compute_vc_citys_sparse(preds, imgs, ref_gt, ref_img, n, ignore_index, sim_thresh):
    L = len(preds)
    if L < n:
        return None
    if ref_gt is None or ref_img is None:
        return None
    if len(imgs) != L:
        return None
    vals = []
    ref_gt_valid = (ref_gt != ignore_index)
    for start in range(0, L - n + 1):
        pred_win = preds[start : start + n]
        img_win = imgs[start : start + n]
        static_all = np.ones(ref_img.shape, dtype=bool)
        for img_t in img_win:
            diff = np.abs(img_t.astype(np.int16) - ref_img.astype(np.int16))
            static_all &= (diff <= sim_thresh)
        m_common = static_all & ref_gt_valid
        denom = int(m_common.sum())
        if denom == 0:
            continue
        pred0 = pred_win[0]
        pred_equal = np.ones(pred0.shape, dtype=bool)
        for p in pred_win[1:]:
            pred_equal &= (p == pred0)
        num = int((m_common & pred_equal & (pred0 == ref_gt)).sum())
        vals.append(num / denom)
    if not vals:
        return None
    return float(np.mean(vals))


def _metrics_from_confusion(confusion):
    confusion = confusion.float()
    intersection = torch.diag(confusion)
    pred_total = confusion.sum(0)
    label_total = confusion.sum(1)
    union = pred_total + label_total - intersection
    iou = intersection / (union + 1e-6)
    acc = intersection / (label_total + 1e-6)
    aacc = intersection.sum() / (confusion.sum() + 1e-6)

    iou_np = iou.cpu().numpy()
    acc_np = acc.cpu().numpy()
    valid_mask = ~(np.isnan(acc_np) | (iou_np == 0))
    if valid_mask.any():
        miou = float(np.nanmean(iou_np[valid_mask]))
        macc = float(np.nanmean(acc_np[valid_mask]))
    else:
        miou = 0.0
        macc = 0.0
    return miou, macc, float(aacc), iou_np, acc_np


def _class_names(dataset, n_classes):
    classes = getattr(dataset, "classes", None)
    if isinstance(classes, dict):
        return [classes.get(i, f"class_{i}") for i in range(n_classes)]
    return [f"class_{i}" for i in range(n_classes)]

def _wait_for_file(path, poll_seconds=30):
    while not os.path.exists(path):
        time.sleep(poll_seconds)


def _touch_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("done\n")


def main(
    config,
    checkpoint_name,
    checkpoint_folder,
    split,
    evaluation,
    best_model,
    write_res,
    write_gif,
    output_subdir=None,
    data_cfg_override=None,
    checkpoint_path=None,
    max_frames=None,
    report_fps=False,
):
    distributed = setup_distributed()
    distributed = distributed or is_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    device = torch.device("cuda", local_rank)
    config = os.path.abspath(os.path.expanduser(config))
    config_dir = os.path.dirname(config)
    with open(config, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    if data_cfg_override:
        cfg["data_cfg"].update(data_cfg_override)
    data_cfg = cfg["data_cfg"]
    image_model_cfg = cfg["image_model_cfg"]
    video_model_cfg = cfg["video_model_cfg"]

    save_dir = cfg["save_dir"]
    if is_main_process():
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        print(f"Distributed: {distributed} | world_size={world_size} | rank={rank} | local_rank={local_rank} | GPUs={visible_gpus}")
    checkpoint = None
    checkpoint_loaded_path = None
    if checkpoint_path:
        checkpoint_loaded_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        if not os.path.isfile(checkpoint_loaded_path):
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_loaded_path}")
        checkpoint = torch.load(checkpoint_loaded_path, map_location="cpu")
    else:
        ckpt_stem = checkpoint_name.split("@")[-1]
        default_dir = os.path.join(save_dir, checkpoint_folder + checkpoint_name)
        default_last = os.path.join(default_dir, ckpt_stem + ".pth.tar")
        default_best = os.path.join(default_dir, "best_model_3_" + ckpt_stem + ".pth.tar")
        alt_last = os.path.join(config_dir, ckpt_stem + ".pth.tar")
        alt_best = os.path.join(config_dir, "best_model_3_" + ckpt_stem + ".pth.tar")
        candidates = [default_best, alt_best] if best_model else [default_last, alt_last]
        for cand in candidates:
            if os.path.exists(cand):
                checkpoint_loaded_path = cand
                checkpoint = torch.load(checkpoint_loaded_path, map_location="cpu")
                if is_main_process():
                    msg = "Loaded best checkpoint at epoch {}".format(checkpoint["epoch"]) if best_model else "Loaded last checkpoint"
                    print(msg)
                break
    if is_main_process() and checkpoint_loaded_path:
        print(f"Using checkpoint: {checkpoint_loaded_path}")

    vis_dir = config_dir
    if output_subdir:
        vis_dir = os.path.join(vis_dir, output_subdir)
    os.makedirs(vis_dir, exist_ok=True)

    # Dataset
    video_dataset, DATASET = prep_infer_image_dataset(data_cfg, split=split)

    single_rank_eval = distributed and len(video_dataset) < world_size
    sync_file = None
    if single_rank_eval:
        run_id = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT") or str(os.getppid())
        sync_file = os.path.join(vis_dir, f".single_rank_eval_done_{run_id}_{split}")
        if is_main_process():
            print(
                f"[eval] Dataset has {len(video_dataset)} videos < world_size={world_size}. "
                "Running single-rank eval to avoid NCCL timeouts."
            )
        if not is_main_process():
            _wait_for_file(sync_file)
            cleanup_distributed()
            return
        distributed = False
        world_size = 1
        rank = 0

    output_root = vis_dir if not distributed else os.path.join(vis_dir, f"rank_{rank}")
    save_folder = os.path.join(output_root, split)
    save_folder_colored = os.path.join(output_root, split + "_colored")
    save_folder_blended = os.path.join(output_root, split + "_blended")
    save_folder_labels = os.path.join(output_root, split + "_labels")
    save_folder_labels_blended = os.path.join(output_root, split + "_labels_blended")
    save_folder_gif = os.path.join(output_root, split + "_gif")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if write_res:
        if not os.path.exists(save_folder_colored):
            os.makedirs(save_folder_colored)
        if not os.path.exists(save_folder_blended):
            os.makedirs(save_folder_blended)
        if not os.path.exists(save_folder_labels):
            os.makedirs(save_folder_labels)
        if not os.path.exists(save_folder_labels_blended):
            os.makedirs(save_folder_labels_blended)
        if write_gif and not os.path.exists(save_folder_gif):
            os.makedirs(save_folder_gif)

    sampler = SequenceDistributedSampler(video_dataset, shuffle=False) if distributed else None
    video_loader = DataLoader(
        video_dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )

    # Trained image model
    image_save_dir = image_model_cfg["image_save_dir"]
    img_checkpoint_folder = image_model_cfg["checkpoint_folder"]
    img_checkpoint_name = image_model_cfg["checkpoint_name"]
    img_best_model = image_model_cfg["best_model"]
    image_cfg_path = _resolve_image_config_path(
        image_save_dir,
        img_checkpoint_folder,
        img_checkpoint_name,
        config_dir,
    )
    if not image_cfg_path:
        raise FileNotFoundError(
            "Image model config not found for '{}'. Checked '{}' and source-domain image roots.".format(
                img_checkpoint_name,
                os.path.join(
                    image_save_dir,
                    img_checkpoint_folder + img_checkpoint_name,
                    img_checkpoint_name.split("@")[-1] + "_config.yaml",
                ),
            )
        )
    with open(image_cfg_path, 'r') as cfg_file:
        image_cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    resolved_dir = os.path.dirname(image_cfg_path)
    resolved_name = os.path.basename(resolved_dir)
    if resolved_name != img_checkpoint_name:
        rel_parent = os.path.relpath(os.path.dirname(resolved_dir), image_cfg.get("save_dir", image_save_dir))
        img_checkpoint_folder = "" if rel_parent == "." else rel_parent + "/"
        img_checkpoint_name = resolved_name
        if is_main_process():
            print(f"Using image model config fallback: {image_cfg_path}")
    seg_model = get_image_model(image_cfg["model_cfg"], DATASET.n_classes)
    seg_model.to(device)
    if img_best_model:
        img_checkpoint = torch.load(os.path.join(image_cfg["save_dir"], img_checkpoint_folder + img_checkpoint_name, "best_model_" + img_checkpoint_name.split("@")[-1] + ".pth.tar"), map_location="cpu")
    else:
        img_checkpoint = torch.load(os.path.join(image_cfg["save_dir"], img_checkpoint_folder + img_checkpoint_name, img_checkpoint_name.split("@")[-1] + ".pth.tar"), map_location="cpu")
    seg_model.load_state_dict(img_checkpoint["model"])

    # Optical flow model
    flow_model = get_flow_model()
    flow_model.to(device)

    # Video model
    model = get_video_model(video_model_cfg, seg_model, flow_model, DATASET.n_classes)
    model.to(device)
    if is_main_process():
        print(f"Model has {sum([p.numel() for p in model.parameters()]):,} parameters")
    if checkpoint is not None:
        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict)

    # Init evaluation metrics
    if evaluation:
        confusion = torch.zeros((DATASET.n_classes, DATASET.n_classes), dtype=torch.int64)
        mvc_sum = {n: 0.0 for n in MVC_WINDOWS}
        mvc_cnt = {n: 0 for n in MVC_WINDOWS}
        is_cityscapes = data_cfg["dataset"].lower() == "cityscapes_seq_corrupt"

    max_frames = max_frames if (max_frames is None or max_frames > 0) else None
    frames_remaining = max_frames
    infer_time = 0.0
    infer_frames = 0
    if report_fps and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Predictions + evaluation
    video_iter = tqdm(video_loader, disable=not is_main_process())
    for (frames_names, frames, labels_names, labels, v_name, homos) in video_iter:
        if frames_remaining is not None and frames_remaining <= 0:
            break
        if frames_remaining is not None:
            keep = min(frames_remaining, len(frames))
            if keep <= 0:
                break
            frames = frames[:keep]
            frames_names = frames_names[:keep]
            homos = homos[:keep]
            frames_remaining -= keep
        frame_list = []
        label_list = []
        pred_list = []

        # Predictions
        frames_for_infer = frames
        if is_main_process() and not report_fps:
            frames_for_infer = tqdm(frames, total=len(frames), leave=False, desc=f"{v_name} infer")
        if report_fps:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            infer_start = time.perf_counter()
        preds = model.infer_video(frames_for_infer, homos, device)
        if report_fps:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            infer_time += time.perf_counter() - infer_start
            infer_frames += len(preds)
        frame_list = [f.numpy() if torch.is_tensor(f) else np.asarray(f) for f in frames]
        label_list = [l.numpy() if torch.is_tensor(l) else np.asarray(l) for l in labels]
        pred_list = [
            p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p) for p in preds
        ]
        label_by_name = {name: labels[i] for i, name in enumerate(labels_names)}

        # Iter over video frames
        frames_iter = frame_list[:-data_cfg["min_vid_len"]] if data_cfg["min_vid_len"] > 0 else frame_list
        if evaluation:
            pred_small_list = []
            gt_small_list = []
            img_gray_list = [] if is_cityscapes else None
            ref_gt_small = None
            ref_img_small = None
        for (i, frame) in enumerate(frames_iter):
            frame_name = frames_names[i]
            label_name = video_dataset.name_to_labelname(frame_name)
            label = label_by_name.get(label_name)
            labeled = label is not None
            label_np = label.detach().cpu().numpy() if labeled and torch.is_tensor(label) else (np.asarray(label) if labeled else None)
            pred = pred_list[i]

            # Compute metrics
            if evaluation and labeled:
                confusion = _update_confusion(confusion, pred, label_np, DATASET.n_classes, DATASET.ignore_index)

            if evaluation:
                pred_small = pred[::MVC_STRIDE, ::MVC_STRIDE]
                if pred_small.dtype != np.uint8:
                    pred_small = pred_small.astype(np.uint8)
                pred_small_list.append(pred_small)
                if labeled:
                    gt_small = label_np[::MVC_STRIDE, ::MVC_STRIDE]
                    if gt_small.dtype != np.uint8:
                        gt_small = gt_small.astype(np.uint8)
                    gt_small_list.append(gt_small)
                    if ref_gt_small is None:
                        ref_gt_small = gt_small
                else:
                    gt_small_list.append(None)

                if is_cityscapes:
                    img_gray = _to_gray_small(frame, pred_small.shape[0], pred_small.shape[1])
                    img_gray_list.append(img_gray)
                    if labeled and ref_img_small is None:
                        ref_img_small = img_gray

            # Save predictions (normal and colored)
            if write_res:
                pred_pil = Image.fromarray(pred_to_mask(pred.copy(), ignore_index=DATASET.ignore_index).astype(np.uint8))
                pred_pil_colored = Image.fromarray(color_predictions(pred, colors=DATASET.colors, ignore_index=DATASET.ignore_index).astype(np.uint8))
                pred_pil_blended = color_predictions(pred, colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame))[1]
                if not os.path.exists(os.path.join(save_folder, v_name)):
                    os.makedirs(os.path.join(save_folder, v_name))
                if not os.path.exists(os.path.join(save_folder, v_name, label_name)):
                    pred_pil.save(os.path.join(save_folder, v_name, label_name))
                if not os.path.exists(os.path.join(save_folder_colored, v_name)):
                    os.makedirs(os.path.join(save_folder_colored, v_name))
                if not os.path.exists(os.path.join(save_folder_colored, v_name, label_name)):
                    pred_pil_colored.save(os.path.join(save_folder_colored, v_name, label_name))
                if not os.path.exists(os.path.join(save_folder_blended, v_name)):
                    os.makedirs(os.path.join(save_folder_blended, v_name))
                if not os.path.exists(os.path.join(save_folder_blended, v_name, label_name)):
                    pred_pil_blended.save(os.path.join(save_folder_blended, v_name, label_name))

                if labeled:
                    label_pil = color_predictions(label_np, colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame))[0]
                    label_pil_blended = color_predictions(label_np, colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame))[1]
                    #if not os.path.exists(os.path.join(save_folder_labels, v_name)):
                    #    os.makedirs(os.path.join(save_folder_labels, v_name))
                    #if not os.path.exists(os.path.join(save_folder_labels, v_name, label_name)):
                    #    label_pil.save(os.path.join(save_folder_labels, v_name, label_name))
                    #if not os.path.exists(os.path.join(save_folder_labels_blended, v_name)):
                    #    os.makedirs(os.path.join(save_folder_labels_blended, v_name))
                    #if not os.path.exists(os.path.join(save_folder_labels_blended, v_name, label_name)):
                    #    label_pil_blended.save(os.path.join(save_folder_labels_blended, v_name, label_name))

        # Save gifs
        if write_res and write_gif:
            if not os.path.exists(os.path.join(save_folder_gif, v_name)):
                os.makedirs(os.path.join(save_folder_gif, v_name))
            frame_gif_list = [Image.fromarray(inverse_normalize(frame)) for frame in frame_list]
            pred_gif_list = [color_predictions(pred, colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame_list[n]))[0] for (n,pred) in enumerate(pred_list)]
            pred_gif_list_blend = [color_predictions(pred, colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame_list[n]))[1] for (n,pred) in enumerate(pred_list)]
            #if not os.path.exists(os.path.join(save_folder_gif, v_name, "frames.gif")):
            #    frame_gif_list[0].save(os.path.join(save_folder_gif, v_name, "frames.gif"), save_all=True, append_images=frame_gif_list[1:], duration=(1000/DATASET.fps), loop=0)
            if not os.path.exists(os.path.join(save_folder_gif, v_name, "preds.gif")):
                pred_gif_list[0].save(os.path.join(save_folder_gif, v_name, "preds.gif"), save_all=True, append_images=pred_gif_list[1:], duration=(1000/DATASET.fps), loop=0)
            if not os.path.exists(os.path.join(save_folder_gif, v_name, "preds_blended.gif")):
                pred_gif_list_blend[0].save(os.path.join(save_folder_gif, v_name, "preds_blended.gif"), save_all=True, append_images=pred_gif_list_blend[1:], duration=(1000/DATASET.fps), loop=0)

            if len(label_list)>1:
                label_gif_list = [color_predictions(label, colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame_list[n]))[0] for (n,label) in enumerate(label_list)]
                label_gif_list_blend = [color_predictions(label, colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame_list[n]))[1] for (n,label) in enumerate(label_list)]
                if not os.path.exists(os.path.join(save_folder_gif, v_name, "labels.gif")):
                    label_gif_list[0].save(os.path.join(save_folder_gif, v_name, "labels.gif"), save_all=True, append_images=label_gif_list[1:], duration=(1000/DATASET.fps), loop=0)
                if not os.path.exists(os.path.join(save_folder_gif, v_name, "labels_blended.gif")):
                    label_gif_list_blend[0].save(os.path.join(save_folder_gif, v_name, "labels_blended.gif"), save_all=True, append_images=label_gif_list_blend[1:], duration=(1000/DATASET.fps), loop=0)

        if evaluation and pred_small_list:
            dense_gt = all(x is not None for x in gt_small_list)
            for n in MVC_WINDOWS:
                if dense_gt:
                    vc_val = _compute_vc_dense(
                        pred_small_list,
                        [x for x in gt_small_list if x is not None],
                        n=n,
                        ignore_index=DATASET.ignore_index,
                    )
                else:
                    if is_cityscapes:
                        vc_val = _compute_vc_citys_sparse(
                            pred_small_list,
                            img_gray_list,
                            ref_gt_small,
                            ref_img_small,
                            n=n,
                            ignore_index=DATASET.ignore_index,
                            sim_thresh=CITYS_SIM_THRESH,
                        )
                    else:
                        vc_val = _compute_vc_sparse_by_valid_windows(
                            pred_small_list,
                            gt_small_list,
                            n=n,
                            ignore_index=DATASET.ignore_index,
                        )
                if vc_val is not None and (not np.isnan(vc_val)):
                    mvc_sum[n] += float(vc_val)
                    mvc_cnt[n] += 1
            if is_main_process():
                video_iter.set_description(f"{v_name}")
    
    if evaluation:
        if distributed:
            confusion = all_reduce_tensor(confusion).cpu()
            mvc_sum_t = torch.tensor([mvc_sum[n] for n in MVC_WINDOWS], device=device, dtype=torch.float64)
            mvc_cnt_t = torch.tensor([mvc_cnt[n] for n in MVC_WINDOWS], device=device, dtype=torch.float64)
            mvc_sum_t = all_reduce_tensor(mvc_sum_t)
            mvc_cnt_t = all_reduce_tensor(mvc_cnt_t)
            for idx, n in enumerate(MVC_WINDOWS):
                mvc_sum[n] = mvc_sum_t[idx].item()
                mvc_cnt[n] = int(mvc_cnt_t[idx].item())

        miou, macc, aacc, per_class_iou, per_class_acc = _metrics_from_confusion(confusion)
        miou_pct = round(miou * 100.0, 2)
        macc_pct = round(macc * 100.0, 2)
        aacc_pct = round(aacc * 100.0, 2)
        per_class_iou_pct = np.nan_to_num(per_class_iou, nan=0.0, posinf=0.0, neginf=0.0) * 100.0
        per_class_acc_pct = np.nan_to_num(per_class_acc, nan=0.0, posinf=0.0, neginf=0.0) * 100.0
        class_names = _class_names(DATASET, len(per_class_iou_pct))
        mvc_metrics = {}
        for n in MVC_WINDOWS:
            if mvc_cnt[n] > 0:
                mvc_metrics[n] = round((mvc_sum[n] / mvc_cnt[n]) * 100.0, 2)
            else:
                mvc_metrics[n] = float("nan")

        if is_main_process():
            mvc8 = mvc_metrics.get(8, float("nan"))
            mvc16 = mvc_metrics.get(16, float("nan"))
            mvc8_str = f"{mvc8:.2f}" if not np.isnan(mvc8) else "nan"
            mvc16_str = f"{mvc16:.2f}" if not np.isnan(mvc16) else "nan"
            print(
                "mIoU = {:.2f} | mAcc = {:.2f} | aAcc = {:.2f} | mVC8 = {} | mVC16 = {}".format(
                    miou_pct, macc_pct, aacc_pct, mvc8_str, mvc16_str
                )
            )
            metrics_name = f"log_metrics_{split}_best_model_3.txt" if best_model else f"log_metrics_{split}.txt"
            with open(os.path.join(vis_dir, metrics_name), "a") as f:
                checkpoint_display = checkpoint_loaded_path or checkpoint_name
                f.write(f"Checkpoint: {checkpoint_display}\n")
                f.write(f"mIoU = {miou_pct:.2f}\n")
                f.write(f"mAcc = {macc_pct:.2f}\n")
                f.write(f"aAcc = {aacc_pct:.2f}\n")
                f.write(f"mVC8 = {mvc8_str} (videos={mvc_cnt.get(8, 0)})\n")
                f.write(f"mVC16 = {mvc16_str} (videos={mvc_cnt.get(16, 0)})\n")
                f.write("per class IoU (%):\n")
                for name, value in zip(class_names, per_class_iou_pct):
                    f.write(f"  {name}: {value:.5f}\n")
                f.write("per class Acc (%):\n")
                for name, value in zip(class_names, per_class_acc_pct):
                    f.write(f"  {name}: {value:.5f}\n")
                f.write("\n")

    if report_fps:
        if distributed:
            fps_stats = torch.tensor([infer_time, float(infer_frames)], device=device, dtype=torch.float64)
            fps_stats = all_reduce_tensor(fps_stats)
            infer_time = float(fps_stats[0].item())
            infer_frames = int(round(fps_stats[1].item()))
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            max_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            max_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        else:
            max_alloc_mb = float("nan")
            max_reserved_mb = float("nan")
        fps = infer_frames / infer_time if infer_time > 0 else float("nan")
        if is_main_process():
            crop_size = data_cfg.get("crop_size")
            crop_desc = f"{crop_size[0]}x{crop_size[1]}" if crop_size else "unknown"
            max_frames_desc = str(max_frames) if max_frames is not None else "all"
            print(
                "FPS (forward only) = {:.2f} | frames = {} | time = {:.3f}s | "
                "input = {} | max_frames = {} | peak_alloc = {:.1f} MiB | peak_reserved = {:.1f} MiB".format(
                    fps,
                    infer_frames,
                    infer_time,
                    crop_desc,
                    max_frames_desc,
                    max_alloc_mb,
                    max_reserved_mb,
                )
            )
            fps_log_name = f"log_fps_{split}.txt"
            with open(os.path.join(vis_dir, fps_log_name), "a") as f:
                f.write(f"Checkpoint: {checkpoint_loaded_path or checkpoint_name}\n")
                f.write(f"FPS_forward_only = {fps:.4f}\n")
                f.write(f"frames = {infer_frames}\n")
                f.write(f"time_sec = {infer_time:.6f}\n")
                f.write(f"input = {crop_desc}\n")
                f.write(f"max_frames = {max_frames_desc}\n")
                f.write(f"peak_alloc_mib = {max_alloc_mb:.2f}\n")
                f.write(f"peak_reserved_mib = {max_reserved_mb:.2f}\n")
                f.write("\n")

    if distributed:
        barrier()
        if is_main_process():
            rank_dirs = [os.path.join(vis_dir, f"rank_{r}") for r in range(world_size)]
            merge_rank_outputs(vis_dir, rank_dirs, cleanup=True)
        barrier()

    if single_rank_eval and is_main_process() and sync_file:
        _touch_file(sync_file)


import argparse
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualization Parameters")
    parser.add_argument("checkpoint_name", metavar="C", type=str, help="Checkpoint to visualize")
    parser.add_argument("--save-dir", required=False, type=str, default="checkpoints",
                         help="Folder where checkpoint (and its config) is located. Should be in config file")
    parser.add_argument("--checkpoint-folder", required=False, type=str, default="", help="Subfolder of checkpoint")
    parser.add_argument("--checkpoint-path", required=False, type=str, default=None,
                        help="Optional explicit checkpoint file path to load")
    parser.add_argument("--split", required=False, type=str, default="val", help="Data split to visualize")
    parser.add_argument("--gpus", required=False, type=str, default=None,
                        help="Comma-separated GPU ids to use, e.g. \"0,1,3\"")
    parser.add_argument("--dataset", required=False, type=str, default=None,
                        help="Override dataset name from config")
    parser.add_argument("--corruption", required=False, type=str, default=None,
                        help="Single corruption name for cityscapes_seq_corrupt")
    parser.add_argument("--corruptions", required=False, type=str, default=None,
                        help="Comma-separated corruptions, e.g. fog,frost,snow,spatter")
    parser.add_argument("--root-images", required=False, type=str, default=None,
                        help="Root folder that contains video subdirectories or sequence folders")
    parser.add_argument("--root-labels", required=False, type=str, default=None,
                        help="Root folder that contains aligned GT subdirectories")
    parser.add_argument("--city-root-images", required=False, type=str, default=None,
                        help="Legacy alias for --root-images on Cityscapes corruptions")
    parser.add_argument("--city-root-labels", required=False, type=str, default=None,
                        help="Legacy alias for --root-labels on Cityscapes corruptions")
    parser.add_argument("--sequence", required=False, type=str, default=None,
                        help="Single sequence/video name to evaluate")
    parser.add_argument("--sequences", required=False, type=str, default=None,
                        help="Comma-separated sequence/video names to evaluate")
    parser.add_argument("--city-seq", required=False, type=str, default=None,
                        help="Legacy alias for --sequence on Cityscapes corruptions")
    parser.add_argument("--city-seqs", required=False, type=str, default=None,
                        help="Legacy alias for --sequences on Cityscapes corruptions")
    parser.add_argument("--frame-folder", required=False, type=str, default=None,
                        help="Optional per-sequence image subfolder name, e.g. Images")
    parser.add_argument("--mask-folder", required=False, type=str, default=None,
                        help="Optional per-sequence label subfolder name, e.g. Labels_classes15")
    parser.add_argument("--img-extension", required=False, type=str, default=None,
                        help="Image file extension, e.g. .jpg or .png")
    parser.add_argument("--label-extension", required=False, type=str, default=None,
                        help="Label file extension, e.g. .png")
    parser.add_argument("--label-suffix", required=False, type=str, default=None,
                        help="Optional label filename suffix before the extension")
    parser.add_argument("--fps", required=False, type=int, default=None,
                        help="Optional dataset FPS used for GIF export")
    parser.add_argument("--output-subdir", required=False, type=str, default=None,
                        help="Optional subdir under checkpoint output directory")
    parser.add_argument("--input-size", nargs=2, type=int, metavar=("H", "W"), default=None,
                        help="Override input resolution as height width, e.g. 1024 2048")
    parser.add_argument("--max-frames", required=False, type=int, default=None,
                        help="Limit total number of frames for inference (across videos)")
    parser.add_argument("--report-fps", dest="report_fps", action="store_true",
                        help="Report forward-only FPS and peak GPU memory usage")
    parser.add_argument('--evaluation', dest='evaluation', action='store_true', help='Compute metrics (default)')
    parser.add_argument('--no-evaluation', dest='evaluation', action='store_false', help='Don\'t compute metrics')
    parser.add_argument('--best-model', dest='best_model', action='store_true', help='Use best checkpoint')
    parser.add_argument('--no-best-model', dest='best_model', action='store_false', help='Use last checkpoint (default)')
    parser.add_argument('--write-res', dest='write_res', action='store_true', help='Write results to disk (default)')
    parser.add_argument('--no-write-res', dest='write_res', action='store_false', help='Do not write results to disk')
    parser.add_argument('--write-gif', dest='write_gif', action='store_true', help='Write GIFs (default)')
    parser.add_argument('--no-gif', dest='write_gif', action='store_false', help='Do not write GIFs')
    parser.add_argument('--metrics-only', dest='metrics_only', action='store_true',
                        help='Compute metrics only (implies --no-write-res)')
    parser.set_defaults(best_model=False)
    parser.set_defaults(evaluation=True)
    parser.set_defaults(write_res=True)
    parser.set_defaults(write_gif=True)
    parser.set_defaults(metrics_only=False)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    save_dir = args.save_dir
    checkpoint_name = args.checkpoint_name
    checkpoint_folder = args.checkpoint_folder
    config = os.path.join(save_dir, checkpoint_folder + checkpoint_name, checkpoint_name.split("@")[-1] + "_config.yaml")
    split = args.split
    evaluation = args.evaluation
    best_model = args.best_model
    write_res = args.write_res
    write_gif = args.write_gif
    if args.metrics_only:
        evaluation = True
        write_res = False
    def _build_data_cfg_override(corruption):
        override = {}
        generic_roots_used = any(
            value is not None
            for value in (
                args.root_images,
                args.root_labels,
                args.frame_folder,
                args.mask_folder,
                args.img_extension,
                args.label_extension,
                args.label_suffix,
                args.sequence,
                args.sequences,
                args.fps,
            )
        )
        if args.dataset:
            override["dataset"] = args.dataset
            override.update(_dataset_defaults(args.dataset))
        elif corruption:
            override["dataset"] = "cityscapes_seq_corrupt"
        elif generic_roots_used:
            override["dataset"] = "folder_sequence"
        if corruption:
            override["corruption"] = corruption
        root_images = args.root_images or args.city_root_images
        root_labels = args.root_labels or args.city_root_labels
        if root_images:
            override["root_images"] = root_images
        if root_labels:
            override["root_labels"] = root_labels
        if args.frame_folder is not None:
            override["frame_folder"] = args.frame_folder
        if args.mask_folder is not None:
            override["mask_folder"] = args.mask_folder
        if args.img_extension:
            override["img_extension"] = args.img_extension
        if args.label_extension:
            override["label_extension"] = args.label_extension
        if args.label_suffix is not None:
            override["label_suffix"] = args.label_suffix
        if args.fps is not None:
            override["fps"] = args.fps
        seqs = []
        for seq_arg in (args.sequence, args.sequences, args.city_seq, args.city_seqs):
            if seq_arg:
                seqs.append(seq_arg)
        if seqs:
            override["sequence_filter"] = seqs
        if args.input_size:
            override["crop_size"] = [args.input_size[0], args.input_size[1]]
            override["square_crop"] = False
        return override or None

    corruptions = []
    if args.corruptions:
        corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    elif args.corruption:
        corruptions = [args.corruption.strip()]

    if corruptions:
        for corruption in corruptions:
            if args.output_subdir:
                if len(corruptions) == 1:
                    output_subdir = args.output_subdir
                else:
                    output_subdir = os.path.join(args.output_subdir, corruption)
            else:
                output_subdir = os.path.join("cityscapes_corruptions", corruption)
            main(
                config,
                checkpoint_name,
                checkpoint_folder,
                split,
                evaluation,
                best_model,
                write_res,
                write_gif,
                output_subdir=output_subdir,
                data_cfg_override=_build_data_cfg_override(corruption),
                checkpoint_path=args.checkpoint_path,
                max_frames=args.max_frames,
                report_fps=args.report_fps,
            )
    else:
        main(
            config,
            checkpoint_name,
            checkpoint_folder,
            split,
            evaluation,
            best_model,
            write_res,
            write_gif,
            output_subdir=args.output_subdir,
            data_cfg_override=_build_data_cfg_override(None),
            checkpoint_path=args.checkpoint_path,
            max_frames=args.max_frames,
            report_fps=args.report_fps,
        )
    cleanup_distributed()
