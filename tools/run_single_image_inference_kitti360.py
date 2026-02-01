#!/usr/bin/env python
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.dataset_utils import parse_datasets, get_transforms
from models.image.models import get_model as get_image_model
from models.opt_flow import get_flow_model
from models.video.models_consistency import get_model as get_video_model
from vis_utils.visualization import color_predictions, inverse_normalize, pred_to_mask

# Editable parameters.
CHECKPOINT_NAME = "29-12_12-52"
CHECKPOINT_FOLDER = "video/"
CHECKPOINT_PATH = "/data1/wangcl/project/SSP/kitti360/video/29-12_12-52/epoch_0010.pth.tar"  # Set to None to use default last checkpoint.
SAVE_DIR = "/data1/wangcl/project/SSP/kitti360"
OUTPUT_DIR = "/home/wangcl/project/SSP/kitti360/video/29-12_12-52/single_img_inference"
IMAGE_PATHS = [
    "/home/wangcl/data/open_video_DGSS/ApolloScape/train/ColorImage/Record008/171206_030550389_Camera_5.jpg",
    "/home/wangcl/data/open_video_DGSS/CamVid/train/images/0001TP/0001TP_006690.png",
    "/home/wangcl/data/open_video_DGSS/cityscapes_sequence/origin_leftImg8bit_sequence/frankfurt/seq1/frankfurt_000000_000275_leftImg8bit.png",
    "/home/wangcl/data/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/fog/frankfurt/seq1/frankfurt_000000_000275_leftImg8bit.png",
    "/home/wangcl/data/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/frost/frankfurt/seq1/frankfurt_000000_000275_leftImg8bit.png",
    "/home/wangcl/data/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/snow/frankfurt/seq1/frankfurt_000000_000275_leftImg8bit.png",
    "/home/wangcl/data/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/spatter/frankfurt/seq1/frankfurt_000000_000275_leftImg8bit.png",
]


def _resolve_image_config_path(image_save_dir, img_checkpoint_folder, img_checkpoint_name, config_dir):
    ckpt_name = img_checkpoint_name.split("@")[-1]
    primary = os.path.join(
        image_save_dir,
        img_checkpoint_folder + img_checkpoint_name,
        f"{ckpt_name}_config.yaml",
    )
    if os.path.isfile(primary):
        return primary

    parent = os.path.dirname(config_dir)
    source_root = os.path.dirname(parent) if os.path.basename(parent) in ("video", "image") else None
    if source_root:
        candidate = os.path.join(source_root, "image", img_checkpoint_name, f"{ckpt_name}_config.yaml")
        if os.path.isfile(candidate):
            return candidate
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


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _load_video_checkpoint(config_dir):
    if CHECKPOINT_PATH:
        checkpoint_loaded_path = os.path.abspath(os.path.expanduser(CHECKPOINT_PATH))
        if not os.path.isfile(checkpoint_loaded_path):
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_loaded_path}")
        return torch.load(checkpoint_loaded_path, map_location="cpu")

    ckpt_stem = CHECKPOINT_NAME.split("@")[-1]
    default_dir = os.path.join(SAVE_DIR, CHECKPOINT_FOLDER + CHECKPOINT_NAME)
    default_last = os.path.join(default_dir, ckpt_stem + ".pth.tar")
    alt_last = os.path.join(config_dir, ckpt_stem + ".pth.tar")
    for cand in (default_last, alt_last):
        if os.path.isfile(cand):
            return torch.load(cand, map_location="cpu")
    raise FileNotFoundError("No checkpoint found. Set CHECKPOINT_PATH explicitly.")


def main():
    ckpt = CHECKPOINT_NAME.split("@")[-1]
    config_path = os.path.join(
        SAVE_DIR,
        CHECKPOINT_FOLDER + CHECKPOINT_NAME,
        f"{ckpt}_config.yaml",
    )
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = _load_yaml(config_path)
    data_cfg = cfg["data_cfg"]
    image_model_cfg = cfg["image_model_cfg"]
    video_model_cfg = cfg["video_model_cfg"]

    dataset = parse_datasets(data_cfg["dataset"], path=data_cfg.get("path"), split="val")[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_save_dir = image_model_cfg["image_save_dir"]
    img_checkpoint_folder = image_model_cfg["checkpoint_folder"]
    img_checkpoint_name = image_model_cfg["checkpoint_name"]
    img_best_model = image_model_cfg["best_model"]
    config_dir = os.path.dirname(config_path)
    image_cfg_path = _resolve_image_config_path(
        image_save_dir,
        img_checkpoint_folder,
        img_checkpoint_name,
        config_dir,
    )
    if not image_cfg_path:
        raise FileNotFoundError("Image model config not found for image checkpoint.")

    image_cfg = _load_yaml(image_cfg_path)
    resolved_dir = os.path.dirname(image_cfg_path)
    resolved_name = os.path.basename(resolved_dir)
    if resolved_name != img_checkpoint_name:
        rel_parent = os.path.relpath(
            os.path.dirname(resolved_dir),
            image_cfg.get("save_dir", image_save_dir),
        )
        img_checkpoint_folder = "" if rel_parent == "." else rel_parent + "/"
        img_checkpoint_name = resolved_name

    seg_model = get_image_model(image_cfg["model_cfg"], dataset.n_classes)
    seg_model.to(device)
    if img_best_model:
        img_ckpt_path = os.path.join(
            image_cfg["save_dir"],
            img_checkpoint_folder + img_checkpoint_name,
            "best_model_" + img_checkpoint_name.split("@")[-1] + ".pth.tar",
        )
    else:
        img_ckpt_path = os.path.join(
            image_cfg["save_dir"],
            img_checkpoint_folder + img_checkpoint_name,
            img_checkpoint_name.split("@")[-1] + ".pth.tar",
        )
    if not os.path.isfile(img_ckpt_path):
        raise FileNotFoundError(f"Image checkpoint not found: {img_ckpt_path}")
    img_checkpoint = torch.load(img_ckpt_path, map_location="cpu")
    seg_model.load_state_dict(img_checkpoint["model"])

    flow_model = get_flow_model()
    flow_model.to(device)

    model = get_video_model(video_model_cfg, seg_model, flow_model, dataset.n_classes)
    model.to(device)

    checkpoint = _load_video_checkpoint(config_dir)
    model.load_state_dict(checkpoint["model"])

    seg_model.eval()
    flow_model.eval()
    model.eval()

    _, _, frame_transforms_val, _ = get_transforms(
        "image",
        data_cfg["crop_size"],
        dataset,
        data_augmentation=False,
        soft_labels=data_cfg.get("soft_labels", False),
        square_crop=data_cfg.get("square_crop", False),
    )

    pred_dir = os.path.join(OUTPUT_DIR, "pred")
    pred_colored_dir = os.path.join(OUTPUT_DIR, "pred_colored")
    pred_blended_dir = os.path.join(OUTPUT_DIR, "pred_blended")
    _ensure_dir(pred_dir)
    _ensure_dir(pred_colored_dir)
    _ensure_dir(pred_blended_dir)

    crop_h, crop_w = data_cfg["crop_size"]

    for img_path in IMAGE_PATHS:
        if not os.path.isfile(img_path):
            print(f"[skip] missing image: {img_path}")
            continue

        orig_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = orig_img.size
        img = orig_img.resize((crop_w, crop_h), Image.BILINEAR)
        img_np = np.array(img)

        transformed = frame_transforms_val(image=img_np)
        frame_tensor = transformed["image"] if isinstance(transformed, dict) else transformed

        with torch.no_grad():
            preds = model.infer_video([frame_tensor], [None], device)
        pred = preds[0]
        pred_np = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
        if (pred_np.shape[0], pred_np.shape[1]) != (orig_h, orig_w):
            pred_np = np.array(
                Image.fromarray(pred_np.astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
            )

        base = os.path.splitext(os.path.basename(img_path))[0] + ".png"

        mask = pred_to_mask(pred_np.copy(), dataset.ignore_index).astype(np.uint8)
        Image.fromarray(mask).save(os.path.join(pred_dir, base))

        colored = color_predictions(
            pred_np.copy(),
            colors=dataset.colors,
            ignore_index=dataset.ignore_index,
        )
        Image.fromarray(colored.astype(np.uint8)).save(os.path.join(pred_colored_dir, base))

        blend_img = np.array(orig_img)
        _, blended = color_predictions(
            pred_np.copy(),
            colors=dataset.colors,
            ignore_index=dataset.ignore_index,
            blend_img=blend_img,
        )
        blended.save(os.path.join(pred_blended_dir, base))

        print(f"[saved] {base}")


if __name__ == "__main__":
    main()
