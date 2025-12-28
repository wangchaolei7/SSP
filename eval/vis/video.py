import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import os
    import PIL.Image as Image
    import numpy as np
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import yaml

    from vis_utils.visualization import color_predictions, inverse_normalize, pred_to_mask
    from utils.metrics import pixel_accuracy, meanIoU, weightedIoU, class_meanIoU, video_consistency, temporal_consistency, temporal_consistency_proposed, temporal_consistency_forwardbackward, get_flow_model, MetricMeter, PerClassMetricMeter
    from models.image.models import get_model as get_image_model
    from models.opt_flow import get_flow_model
    from models.video.models_consistency import get_model as get_video_model
    from data.dataset_prep import prep_infer_image_dataset
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
        DistributedEvalSampler,
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

def _iou_from_confusion(confusion):
    intersection = torch.diag(confusion).float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - intersection
    iou = intersection / (union + 1e-6)
    return iou, union

def main(config, checkpoint_name, checkpoint_folder, split, evaluation, best_model, write_res, output_subdir=None, data_cfg_override=None):
    distributed = setup_distributed()
    distributed = distributed or is_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    device = torch.device("cuda", local_rank)
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
    if os.path.exists(os.path.join(save_dir, checkpoint_folder + checkpoint_name, checkpoint_name.split("@")[-1] + ".pth.tar")):
        if best_model:
            checkpoint = torch.load(os.path.join(save_dir, checkpoint_folder + checkpoint_name, "best_model_3_" + checkpoint_name.split("@")[-1] + ".pth.tar"), map_location="cpu")
            if is_main_process():
                print("Loaded best checkpoint at epoch {}".format(checkpoint["epoch"]))
        else:
            checkpoint = torch.load(os.path.join(save_dir, checkpoint_folder + checkpoint_name, checkpoint_name.split("@")[-1] + ".pth.tar"), map_location="cpu")
            if is_main_process():
                print("Loaded last checkpoint")
    else:
        checkpoint = None

    vis_dir = os.path.join(cfg["save_dir"], checkpoint_name)
    if output_subdir:
        vis_dir = os.path.join(vis_dir, output_subdir)
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
        if not os.path.exists(save_folder_gif):
            os.makedirs(save_folder_gif)

    # Dataset 
    video_dataset, DATASET = prep_infer_image_dataset(data_cfg, split=split)
    sampler = DistributedEvalSampler(video_dataset, shuffle=False) if distributed else None
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
    with open(os.path.join(image_save_dir, img_checkpoint_folder + img_checkpoint_name, img_checkpoint_name.split("@")[-1] + "_config.yaml"), 'r') as cfg_file:
        image_cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
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
        # Flow prediction model for TC
        model_raft = get_flow_model()
        model_raft = model_raft.to(device)

        mIoU1 = MetricMeter()
        mIoU2 = MetricMeter()
        wIoU = MetricMeter()
        accuracy = MetricMeter()
        tc = MetricMeter()
        # tc_p = MetricMeter()
        tc_fb = MetricMeter()
        n_frames_vc = DATASET.n_frames_vc
        vc = []
        classes_mIoU = PerClassMetricMeter(DATASET.n_classes)
        confusion = torch.zeros((DATASET.n_classes, DATASET.n_classes), dtype=torch.int64)

    # Predictions + evaluation
    video_iter = tqdm(video_loader, disable=not is_main_process())
    for (frames_names, frames, labels_names, labels, v_name, homos) in video_iter:
        frame_list = []
        label_list = []
        pred_list = []
        
        # Predictions
        preds = model.infer_video(frames, homos, device)
        frame_list = [f.numpy() for f in frames]
        label_list = [l.numpy() for l in labels]
        pred_list = [p.numpy() for p in preds]

        # Iter over video frames
        frames_iter = frame_list[:-data_cfg["min_vid_len"]] if data_cfg["min_vid_len"] > 0 else frame_list
        for (i, frame) in enumerate(frames_iter):
            frame_name = frames_names[i]
            label_name = video_dataset.name_to_labelname(frame_name)
            labeled = label_name in labels_names
            if labeled:
                label = labels[labels_names.index(label_name)]
            pred = preds[i]

            # Compute metrics
            if evaluation and labeled:
                mIoU1.update(meanIoU(pred, label, DATASET.n_classes, ignore_index=DATASET.ignore_index))
                valid_classes = torch.unique(label[label<255])
                valid_classes = np.array(torch.nn.functional.one_hot(valid_classes, num_classes=DATASET.n_classes).sum(0))
                classes_mIoU.update(np.array(class_meanIoU(pred, label, DATASET.n_classes, ignore_index=DATASET.ignore_index)), valid_classes)
                mIoU2.update(classes_mIoU.last_values.sum()/valid_classes.astype(float).sum())
                wIoU.update(weightedIoU(pred, label, DATASET.n_classes, ignore_index=DATASET.ignore_index))
                accuracy.update(pixel_accuracy(pred, label))
                confusion = _update_confusion(confusion, pred, label, DATASET.n_classes, DATASET.ignore_index)

            # Save predictions (normal and colored)
            if write_res:
                pred_pil = Image.fromarray(pred_to_mask(pred.numpy(), ignore_index=DATASET.ignore_index).astype(np.uint8))
                pred_pil_colored = Image.fromarray(color_predictions(pred.numpy(), colors=DATASET.colors, ignore_index=DATASET.ignore_index).astype(np.uint8))
                pred_pil_blended = color_predictions(pred.numpy(), colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame))[1]
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
                    label_pil = color_predictions(label.numpy(), colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame))[0]
                    label_pil_blended = color_predictions(label.numpy(), colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame))[1]
                    #if not os.path.exists(os.path.join(save_folder_labels, v_name)):
                    #    os.makedirs(os.path.join(save_folder_labels, v_name))
                    #if not os.path.exists(os.path.join(save_folder_labels, v_name, label_name)):
                    #    label_pil.save(os.path.join(save_folder_labels, v_name, label_name))
                    #if not os.path.exists(os.path.join(save_folder_labels_blended, v_name)):
                    #    os.makedirs(os.path.join(save_folder_labels_blended, v_name))
                    #if not os.path.exists(os.path.join(save_folder_labels_blended, v_name, label_name)):
                    #    label_pil_blended.save(os.path.join(save_folder_labels_blended, v_name, label_name))

        # Save gifs
        if write_res:
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

        if evaluation:
            if len(frame_list) <= 1:
                continue
            tc.update(temporal_consistency(frame_list, pred_list, model_raft, DATASET.n_classes, device))
            # tc_p.update(temporal_consistency_proposed(frame_list, pred_list, model_raft, DATASET.n_classes, device))
            #tc_fb.update(temporal_consistency_forwardbackward(frame_list, pred_list, model_raft, DATASET.n_classes, device))
            if len(label_list) > n_frames_vc and len(label_list) == len(pred_list):
                vc.extend(video_consistency(label_list, pred_list, n_frames_vc))
            if is_main_process():
                video_iter.set_description(f"{v_name}: TC = {tc.avg:.4f}")
    
    if evaluation:
        if distributed:
            confusion = all_reduce_tensor(confusion).cpu()

        iou, union = _iou_from_confusion(confusion)
        valid = union > 0
        global_miou = iou[valid].mean().item() if valid.any() else 0.0
        global_per_classes_mIoU = {v: iou[k-1].item() for (k, v) in DATASET.classes.items() if k > 0} if DATASET.ignore_index > 0 else {v: iou[k].item() for (k, v) in DATASET.classes.items()}

        # tc_sum = torch.tensor(tc.sum, device=device)
        tc_sum = tc.sum.to(device) if torch.is_tensor(tc.sum) else torch.as_tensor(tc.sum, device=device)
        tc_count = torch.tensor(tc.count, device=device)
        if distributed:
            tc_sum = all_reduce_tensor(tc_sum)
            tc_count = all_reduce_tensor(tc_count)
        tc_avg = (tc_sum / tc_count.clamp(min=1)).item()

        if is_main_process():
            print(f"mIoU = {global_miou:.4f} | Temporal Consistency = {tc_avg:.4f}")
            print(global_per_classes_mIoU)
            metrics_name = f"log_metrics_{split}_best_model_3.txt" if best_model else f"log_metrics_{split}.txt"
            with open(os.path.join(vis_dir, metrics_name), "a") as f:
                if checkpoint is not None:
                    f.write("Checkpoint {} at epoch {}\n".format(checkpoint_name, checkpoint["epoch"]))
                f.write(f"mIoU = {global_miou:.4f}\n")
                f.write(f"Temporal Consistency = {tc_avg:.4f}\n")
                f.write(f"per class mIoU:\n")
                for (k, v) in global_per_classes_mIoU.items():
                    f.write(f"  {k}: {v:.5f}\n")

    if distributed:
        barrier()
        if is_main_process():
            rank_dirs = [os.path.join(vis_dir, f"rank_{r}") for r in range(world_size)]
            merge_rank_outputs(vis_dir, rank_dirs, cleanup=True)
        barrier()


import argparse
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualization Parameters")
    parser.add_argument("checkpoint_name", metavar="C", type=str, help="Checkpoint to visualize")
    parser.add_argument("--save-dir", required=False, type=str, default="checkpoints",
                         help="Folder where checkpoint (and its config) is located. Should be in config file")
    parser.add_argument("--checkpoint-folder", required=False, type=str, default="", help="Subfolder of checkpoint")
    parser.add_argument("--split", required=False, type=str, default="val", help="Data split to visualize")
    parser.add_argument("--gpus", required=False, type=str, default=None,
                        help="Comma-separated GPU ids to use, e.g. \"0,1,3\"")
    parser.add_argument("--dataset", required=False, type=str, default=None,
                        help="Override dataset name from config")
    parser.add_argument("--corruption", required=False, type=str, default=None,
                        help="Single corruption name for cityscapes_seq_corrupt")
    parser.add_argument("--corruptions", required=False, type=str, default=None,
                        help="Comma-separated corruptions, e.g. fog,frost,snow,spatter")
    parser.add_argument("--city-root-images", required=False, type=str, default=None,
                        help="Root folder for Cityscapes corrupted images")
    parser.add_argument("--city-root-labels", required=False, type=str, default=None,
                        help="Root folder for Cityscapes gtFine labels")
    parser.add_argument("--output-subdir", required=False, type=str, default=None,
                        help="Optional subdir under checkpoint output directory")
    parser.add_argument('--evaluation', dest='evaluation', action='store_true', help='Compute metrics (default)')
    parser.add_argument('--no-evaluation', dest='evaluation', action='store_false', help='Don\'t compute metrics')
    parser.add_argument('--best-model', dest='best_model', action='store_true', help='Use best checkpoint')
    parser.add_argument('--no-best-model', dest='best_model', action='store_false', help='Use last checkpoint (default)')
    parser.add_argument('--write-res', dest='write_res', action='store_true', help='Write results to disk (default)')
    parser.add_argument('--no-write-res', dest='write_res', action='store_false', help='Do not write results to disk')
    parser.set_defaults(best_model=False)
    parser.set_defaults(evaluation=True)
    parser.set_defaults(write_res=True)
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
    def _build_data_cfg_override(corruption):
        override = {}
        if args.dataset:
            override["dataset"] = args.dataset
        elif corruption:
            override["dataset"] = "cityscapes_seq_corrupt"
        if corruption:
            override["corruption"] = corruption
        if args.city_root_images:
            override["root_images"] = args.city_root_images
        if args.city_root_labels:
            override["root_labels"] = args.city_root_labels
        return override or None

    corruptions = []
    if args.corruptions:
        corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    elif args.corruption:
        corruptions = [args.corruption.strip()]

    if corruptions:
        for corruption in corruptions:
            output_subdir = os.path.join("cityscapes_corruptions", corruption)
            main(
                config,
                checkpoint_name,
                checkpoint_folder,
                split,
                evaluation,
                best_model,
                write_res,
                output_subdir=output_subdir,
                data_cfg_override=_build_data_cfg_override(corruption),
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
            output_subdir=args.output_subdir,
            data_cfg_override=_build_data_cfg_override(None),
        )
    cleanup_distributed()
