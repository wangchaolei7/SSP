import numpy as np
from utils.distributed import is_main_process
from data.image.image_dataset import ImageDataset, ImageInferenceDataset, ImageLogitsDataset
from data.video.video_dataset import VideoDataset, VideoLogitsDataset
from data.apolloscape_dataset import (
    ApolloScapeImageDataset,
    ApolloScapeImageInferenceDataset,
    ApolloScapeImageLogitsDataset,
    ApolloScapeVideoDataset,
    ApolloScapeVideoLogitsDataset,
)
from data.kitti360_dataset import (
    Kitti360ImageDataset,
    Kitti360ImageInferenceDataset,
    Kitti360ImageLogitsDataset,
    Kitti360VideoDataset,
    Kitti360VideoLogitsDataset,
)
from data.dataset_utils import parse_datasets, get_transforms
from data.cityscapes_sequence_dataset import (
    CityscapesSequenceCorruptionsDataset,
    CITYSCAPES_SEQ_CORRUPT,
)


def _validate_num_classes(data_cfg, DATASET):
    cfg_num_classes = data_cfg.get("num_classes")
    if cfg_num_classes is not None and cfg_num_classes != DATASET.n_classes:
        raise ValueError(
            f"num_classes mismatch: config has {cfg_num_classes}, DATASET has {DATASET.n_classes}"
        )


def prep_video_dataset(data_cfg):
    DATASET, data_folder, video_train_idx, video_val_idx = parse_datasets(data_cfg["dataset"], path=data_cfg["path"])
    _validate_num_classes(data_cfg, DATASET)

    augmentations, frame_transforms, frame_transforms_val, mask_transforms = get_transforms("video", 
                                                                                            data_cfg["crop_size"], 
                                                                                            DATASET, 
                                                                                            data_augmentation=data_cfg.get("data_augmentation", False),
                                                                                            soft_labels=data_cfg.get("soft_labels", False),
                                                                                            square_crop=data_cfg.get("square_crop", False))

    dataset_name = data_cfg["dataset"].lower()
    if dataset_name == "apolloscape":
        traindatasetclass = ApolloScapeVideoLogitsDataset if data_cfg.get("logit_distillation", False) else ApolloScapeVideoDataset
        train_dataset = traindatasetclass(
            data_folder,
            video_train_idx,
            DATASET,
            data_cfg,
            split="train",
            training=True,
            joint_transforms=augmentations,
            img_transforms=frame_transforms,
            segmentation_transforms=mask_transforms,
        )
        val_dataset = ApolloScapeVideoDataset(
            data_folder,
            video_val_idx,
            DATASET,
            data_cfg,
            split="val",
            training=False,
            img_transforms=frame_transforms_val,
            segmentation_transforms=mask_transforms,
        )
    elif dataset_name == "kitti360":
        traindatasetclass = Kitti360VideoLogitsDataset if data_cfg.get("logit_distillation", False) else Kitti360VideoDataset
        train_dataset = traindatasetclass(
            data_folder,
            video_train_idx,
            DATASET,
            data_cfg,
            split="train",
            training=True,
            joint_transforms=augmentations,
            img_transforms=frame_transforms,
            segmentation_transforms=mask_transforms,
        )
        val_dataset = Kitti360VideoDataset(
            data_folder,
            video_val_idx,
            DATASET,
            data_cfg,
            split="val",
            training=False,
            img_transforms=frame_transforms_val,
            segmentation_transforms=mask_transforms,
        )
    else:
        traindatasetclass = VideoLogitsDataset if data_cfg.get("logit_distillation", False) else VideoDataset
        train_dataset = traindatasetclass(
            data_folder,
            video_train_idx,
            DATASET,
            data_cfg,
            training=True,
            joint_transforms=augmentations,
            img_transforms=frame_transforms,
            segmentation_transforms=mask_transforms,
        )
        val_dataset = VideoDataset(
            data_folder,
            video_val_idx,
            DATASET,
            data_cfg,
            training=False,
            img_transforms=frame_transforms_val,
            segmentation_transforms=mask_transforms,
        )

    if is_main_process():
        print(f"Train dataset: {len(train_dataset)} samples, Validation dataset: {len(val_dataset)} samples")
    return train_dataset, val_dataset, DATASET


def prep_image_dataset(data_cfg):
    DATASET, data_folder, video_train_idx, video_val_idx = parse_datasets(data_cfg["dataset"], path=data_cfg["path"])
    _validate_num_classes(data_cfg, DATASET)

    augmentations, frame_transforms, frame_transforms_val, mask_transforms = get_transforms("image", 
                                                                                            data_cfg["crop_size"], 
                                                                                            DATASET, 
                                                                                            data_augmentation=data_cfg.get("data_augmentation", False),
                                                                                            soft_labels=data_cfg.get("soft_labels", False),
                                                                                            square_crop=data_cfg.get("square_crop", False))

    dataset_name = data_cfg["dataset"].lower()
    if dataset_name == "apolloscape":
        traindatasetclass = ApolloScapeImageLogitsDataset if data_cfg.get("logit_distillation", False) else ApolloScapeImageDataset
        train_dataset = traindatasetclass(
            data_folder,
            video_train_idx,
            DATASET,
            data_cfg,
            split="train",
            training=True,
            joint_transforms=augmentations,
            img_transforms=frame_transforms,
            segmentation_transforms=mask_transforms,
        )

        val_dataset = ApolloScapeImageDataset(
            data_folder,
            video_val_idx,
            DATASET,
            data_cfg,
            split="val",
            training=False,
            joint_transforms=None,
            img_transforms=frame_transforms_val,
            segmentation_transforms=mask_transforms,
        )
    elif dataset_name == "kitti360":
        traindatasetclass = Kitti360ImageLogitsDataset if data_cfg.get("logit_distillation", False) else Kitti360ImageDataset
        train_dataset = traindatasetclass(
            data_folder,
            video_train_idx,
            DATASET,
            data_cfg,
            split="train",
            training=True,
            joint_transforms=augmentations,
            img_transforms=frame_transforms,
            segmentation_transforms=mask_transforms,
        )

        val_dataset = Kitti360ImageDataset(
            data_folder,
            video_val_idx,
            DATASET,
            data_cfg,
            split="val",
            training=False,
            joint_transforms=None,
            img_transforms=frame_transforms_val,
            segmentation_transforms=mask_transforms,
        )
    else:
        traindatasetclass = ImageLogitsDataset if data_cfg.get("logit_distillation", False) else ImageDataset
        train_dataset = traindatasetclass(
            data_folder,
            video_train_idx,
            DATASET,
            data_cfg,
            training=True,
            joint_transforms=augmentations,
            img_transforms=frame_transforms,
            segmentation_transforms=mask_transforms,
        )

        val_dataset = ImageDataset(
            data_folder,
            video_val_idx,
            DATASET,
            data_cfg,
            training=False,
            joint_transforms=None,
            img_transforms=frame_transforms_val,
            segmentation_transforms=mask_transforms,
        )

    if is_main_process():
        print(f"Train dataset contains {len(train_dataset)} samples")
        print(f"Validation dataset contains {len(val_dataset)} samples")

    return train_dataset, val_dataset, DATASET



def prep_infer_image_dataset(data_cfg, split="val", val_skip_frames=1):
    dataset_name = data_cfg["dataset"].lower()
    if dataset_name == "cityscapes_seq_corrupt":
        DATASET = CITYSCAPES_SEQ_CORRUPT
        _validate_num_classes(data_cfg, DATASET)
        root_images = data_cfg.get("root_images")
        root_labels = data_cfg.get("root_labels")
        corruption = data_cfg.get("corruption")
        if not root_images or not root_labels:
            raise ValueError("cityscapes_seq_corrupt requires data_cfg.root_images and data_cfg.root_labels")
        if not corruption:
            raise ValueError("cityscapes_seq_corrupt requires data_cfg.corruption")
        _, _, frame_transforms, mask_transforms = get_transforms(
            "image", data_cfg["crop_size"], DATASET, data_augmentation=False
        )
        video_dataset = CityscapesSequenceCorruptionsDataset(
            root_images,
            root_labels,
            corruption,
            DATASET,
            data_cfg,
            split=split,
            img_transforms=frame_transforms,
            segmentation_transforms=mask_transforms,
            val_skip_frames=val_skip_frames,
            log_stats=is_main_process(),
        )
    else:
        DATASET, data_folder, video_train_idx, video_val_idx = parse_datasets(
            data_cfg["dataset"], path=data_cfg["path"], split=split
        )
        video_indices = video_train_idx if split == "train" else video_val_idx
        _validate_num_classes(data_cfg, DATASET)
        _, _, frame_transforms, mask_transforms = get_transforms(
            "image", data_cfg["crop_size"], DATASET, data_augmentation=False
        )
        if dataset_name == "apolloscape":
            video_dataset = ApolloScapeImageInferenceDataset(
                data_folder,
                video_indices,
                DATASET,
                data_cfg,
                split=split,
                img_transforms=frame_transforms,
                segmentation_transforms=mask_transforms,
                val_skip_frames=val_skip_frames,
            )
        elif dataset_name == "kitti360":
            video_dataset = Kitti360ImageInferenceDataset(
                data_folder,
                video_indices,
                DATASET,
                data_cfg,
                split=split,
                img_transforms=frame_transforms,
                segmentation_transforms=mask_transforms,
                val_skip_frames=val_skip_frames,
            )
        else:
            video_dataset = ImageInferenceDataset(
                data_folder,
                video_indices,
                DATASET,
                data_cfg,
                img_transforms=frame_transforms,
                segmentation_transforms=mask_transforms,
                val_skip_frames=val_skip_frames,
            )

    if is_main_process():
        print(f"Inference dataset contains {len(video_dataset)} videos")

    return video_dataset, DATASET


def prep_infer_video_dataset(data_cfg, split="val", val_skip_frames=1):
    return prep_infer_image_dataset(data_cfg, split=split, val_skip_frames=val_skip_frames)
