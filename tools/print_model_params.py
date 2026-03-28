#!/usr/bin/env python3
import argparse
import os
import sys
import yaml

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import datasets as dataset_defs
from models.image.models import get_model as get_image_model
from models.opt_flow import get_flow_model
from models.video.models_consistency import get_model as get_video_model


DATASET_MAP = {
    "uavid": dataset_defs.UAVID,
    "ruralscapes": dataset_defs.RURALSCAPES,
    "apolloscape": dataset_defs.APOLLOSCAPE,
    "kitti360": dataset_defs.KITTI360,
    "camvid": dataset_defs.CAMVID,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build SSP model and print parameter counts")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to config file (e.g. config/image/base_camvid.yaml or config/video/ssp_camvid.yaml)",
    )
    parser.add_argument(
        "--task",
        choices=["image", "video"],
        default=None,
        help="Force task type; if omitted, inferred from config contents",
    )
    return parser.parse_args()


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_config_path(config, default_subdir):
    if os.path.isfile(config):
        return config
    repo_root = _repo_root()
    candidate = os.path.join(repo_root, config)
    if os.path.isfile(candidate):
        return candidate
    candidate = os.path.join(repo_root, default_subdir, config)
    if os.path.isfile(candidate):
        return candidate
    tried = [
        os.path.abspath(config),
        os.path.join(repo_root, config),
        os.path.join(repo_root, default_subdir, config),
    ]
    raise FileNotFoundError(f"Config file not found. Tried: {tried}")


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _get_num_classes(data_cfg):
    if data_cfg is None:
        raise ValueError("Missing data_cfg in config")
    if data_cfg.get("num_classes") is not None:
        return int(data_cfg["num_classes"])
    dataset_name = (data_cfg.get("dataset") or "").lower()
    if not dataset_name:
        raise ValueError("data_cfg.dataset is required when num_classes is not set")
    dataset_cls = DATASET_MAP.get(dataset_name)
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Please set data_cfg.num_classes explicitly.")
    return int(dataset_cls.n_classes)


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _resolve_image_config_path(image_model_cfg):
    image_save_dir = image_model_cfg["image_save_dir"]
    checkpoint_folder = image_model_cfg.get("checkpoint_folder", "")
    checkpoint_name = image_model_cfg["checkpoint_name"]
    config_name = checkpoint_name.split("@")[-1] + "_config.yaml"
    config_path = os.path.join(image_save_dir, checkpoint_folder + checkpoint_name, config_name)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Image model config not found: {config_path}")
    return config_path


def _infer_task(config_path, cfg):
    if "video_model_cfg" in cfg or "image_model_cfg" in cfg:
        return "video"
    if "model_cfg" in cfg:
        return "image"
    if os.path.sep + "video" + os.path.sep in config_path:
        return "video"
    if os.path.sep + "image" + os.path.sep in config_path:
        return "image"
    raise ValueError("Cannot infer task type. Please pass --task image|video.")


def build_image_model(cfg):
    data_cfg = cfg["data_cfg"]
    model_cfg = cfg["model_cfg"]
    n_classes = _get_num_classes(data_cfg)
    model = get_image_model(model_cfg, n_classes)
    return model, n_classes


def build_video_model(cfg):
    data_cfg = cfg["data_cfg"]
    n_classes = _get_num_classes(data_cfg)
    image_cfg_path = _resolve_image_config_path(cfg["image_model_cfg"])
    image_cfg = _load_yaml(image_cfg_path)
    seg_model = get_image_model(image_cfg["model_cfg"], n_classes)
    flow_model = get_flow_model()
    model = get_video_model(cfg["video_model_cfg"], seg_model, flow_model, n_classes)
    return model, n_classes, image_cfg_path


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    if args.task is None:
        try:
            config_path = _resolve_config_path(args.config, "config/image")
        except FileNotFoundError:
            config_path = _resolve_config_path(args.config, "config/video")
    else:
        default_subdir = "config/video" if args.task == "video" else "config/image"
        config_path = _resolve_config_path(args.config, default_subdir)
    cfg = _load_yaml(config_path)
    task = args.task or _infer_task(config_path, cfg)

    if task == "image":
        model, n_classes = build_image_model(cfg)
        model.to("cpu")
        total, trainable = _count_params(model)
        print(f"Task: image")
        print(f"Config: {config_path}")
        print(f"Num classes: {n_classes}")
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,}")
        return

    if task == "video":
        model, n_classes, image_cfg_path = build_video_model(cfg)
        model.to("cpu")
        total, trainable = _count_params(model)
        print(f"Task: video")
        print(f"Config: {config_path}")
        print(f"Image model config: {image_cfg_path}")
        print(f"Num classes: {n_classes}")
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,}")
        return

    raise ValueError(f"Unsupported task: {task}")


if __name__ == "__main__":
    main()
