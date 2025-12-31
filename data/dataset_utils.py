import os
from data.datasets import UAVID, RURALSCAPES, APOLLOSCAPE, KITTI360, CAMVID

import cv2
import data.utils.images_transforms as image_transforms
import albumentations as aug
from albumentations.pytorch.transforms import ToTensorV2


def parse_datasets(name, path=None, split="train"):
    name = name.lower()
    DATASET = None
    if name == "uavid":
        DATASET = UAVID
    elif name == "ruralscapes":
        DATASET = RURALSCAPES
    elif name == "apolloscape":
        DATASET = APOLLOSCAPE
    elif name == "kitti360":
        DATASET = KITTI360
    elif name == "camvid":
        DATASET = CAMVID
    assert DATASET is not None, f"Dataset {name} not implemented"

    if path is None:
        path = DATASET.path

    if name == "apolloscape":
        data_folder = path

        def list_records(split_name):
            frame_root = os.path.join(path, split_name, DATASET.frame_folder)
            if not os.path.isdir(frame_root):
                raise FileNotFoundError(f"ApolloScape split not found: {frame_root}")
            records = [d for d in os.listdir(frame_root) if os.path.isdir(os.path.join(frame_root, d))]
            records.sort()
            return records

        if split == "test":
            test_split = "test" if os.path.isdir(os.path.join(path, "test")) else "val"
            video_train_idx = None
            video_val_idx = list_records(test_split)
        else:
            video_train_idx = list_records("train")
            video_val_idx = list_records("val")
        return DATASET, data_folder, video_train_idx, video_val_idx

    if name == "kitti360":
        data_folder = path

        def list_records(split_name):
            label_root = os.path.join(path, split_name, "data_2d_semantics")
            if not os.path.isdir(label_root):
                raise FileNotFoundError(f"KITTI-360 split not found: {label_root}")
            records = []
            for d in os.listdir(label_root):
                label_dir = os.path.join(label_root, d, "image_00", DATASET.mask_folder)
                if os.path.isdir(label_dir):
                    records.append(d)
            records.sort()
            return records

        if split == "test":
            test_split = "test" if os.path.isdir(os.path.join(path, "test")) else "val"
            video_train_idx = None
            video_val_idx = list_records(test_split)
        else:
            video_train_idx = list_records("train")
            video_val_idx = list_records("val")
        return DATASET, data_folder, video_train_idx, video_val_idx

    if name == "camvid":
        data_folder = path

        def list_records(split_name):
            label_root = os.path.join(path, split_name, DATASET.mask_folder)
            if not os.path.isdir(label_root):
                raise FileNotFoundError(f"CamVid split not found: {label_root}")
            records = [d for d in os.listdir(label_root) if os.path.isdir(os.path.join(label_root, d))]
            records.sort()
            return records

        if split == "test":
            test_split = "test" if os.path.isdir(os.path.join(path, "test")) else "val"
            video_train_idx = None
            video_val_idx = list_records(test_split)
        else:
            video_train_idx = list_records("train")
            video_val_idx = list_records("val")
        return DATASET, data_folder, video_train_idx, video_val_idx

    data_folder = os.path.join(path, "data")

    def get_split_indices(split):
        with open(os.path.join(path, split)) as f:
            indices = f.readlines(-1)

        video_indices = []
        for idx in indices:
            v_idx = idx[:-1] # remove \n
            if v_idx not in video_indices:
                video_indices.append(v_idx)

        return video_indices

    if split == "test":
        video_train_idx = None
        video_val_idx = get_split_indices("test.txt")
    else:
        video_train_idx = get_split_indices("train.txt")
        video_val_idx = get_split_indices("val.txt")

    return DATASET, data_folder, video_train_idx, video_val_idx


def get_transforms(data_type, crop_size, DATASET, data_augmentation=False, soft_labels=False, square_crop=False):
    normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if data_type == "video":
        frame_transforms_val = image_transforms.VideoCompose([
                image_transforms.VideoCenterCrop((crop_size[0], crop_size[0])) if square_crop else image_transforms.VideoIdentity(),
                image_transforms.VideoNormalize(mean=normalize[0], std=normalize[1]),
                image_transforms.VideoToTensor(),
            ])

        mask_transforms = image_transforms.Compose([
                image_transforms.CenterCrop((crop_size[0], crop_size[0])) if square_crop else image_transforms.Identity(),
                image_transforms.LabelToTensor(n_classes=DATASET.n_classes, convert_labels=DATASET.convert_labels) if not soft_labels else image_transforms.ToTensor()
            ])
            
        if data_augmentation:
            augmentations = image_transforms.VideoCompose_wLabel([
                    image_transforms.VideoRandomHorizontalFlip(0.5),
                    image_transforms.VideoRandomScaleCrop(0.75, crop_size, scales=[1.1,1.15,1.2,1.25,1.3,1.35,1.4], interpolation="bilinear"),
                    image_transforms.VideoRandomCrop(1, crop_size[0]) if square_crop else image_transforms.VideoIdentity_wLabel()
                ])

            frame_transforms = image_transforms.VideoCompose([
                    image_transforms.VideoColorJitter(0.3),
                    image_transforms.VideoCenterCrop((crop_size[0], crop_size[0])) if square_crop else image_transforms.VideoIdentity(),
                    image_transforms.VideoNormalize(mean=normalize[0], std=normalize[1]),
                    image_transforms.VideoToTensor(),
                ])
        else:
            augmentations = None
            frame_transforms = frame_transforms_val

    elif data_type == "image":
        transform_aug = aug.Compose(
            [
                aug.HorizontalFlip(p=0.5),
                aug.OneOf(
                    [
                        aug.Perspective(p=1, scale=(0.01, 0.05)),
                        aug.ShiftScaleRotate(
                            scale_limit=0.1,
                            rotate_limit=15,
                            shift_limit=0.0625,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=0,
                            value=0,
                            mask_value=0,
                            p=1.,
                        ),
                    ],
                    p=0.2
                ),
                aug.OneOf(
                    [
                        aug.RandomBrightnessContrast(p=0.5),
                        aug.HueSaturationValue(p=0.5),
                        aug.RandomGamma(p=0.5),
                        aug.CLAHE(p=0.5),
                    ],
                    p=0.8,
                ),
                aug.OneOf(
                    [
                        aug.ISONoise(p=0.5),
                        aug.GaussNoise(p=0.5),
                        aug.ImageCompression(p=0.5),
                        aug.Sharpen(p=0.5),
                    ],
                    p=0.6,
                ),
                aug.RandomCrop(crop_size[0], crop_size[0]) if square_crop else aug.NoOp(),
            ],
            is_check_shapes=False)

        augmentations = transform_aug if data_augmentation else None

        normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        frame_transforms = aug.Compose([
            aug.CenterCrop(crop_size[0], crop_size[0]) if square_crop else aug.NoOp(),
            aug.Normalize(mean=normalize[0], std=normalize[1], max_pixel_value=255.),
            ToTensorV2(),
        ])
        frame_transforms_val = frame_transforms

        mask_transforms = image_transforms.Compose([
            image_transforms.CenterCrop((crop_size[0], crop_size[0])) if square_crop else image_transforms.Identity(),
            image_transforms.LabelToTensor(n_classes=DATASET.n_classes, convert_labels=DATASET.convert_labels) if not soft_labels else image_transforms.ToTensor()
        ])

    return augmentations, frame_transforms, frame_transforms_val, mask_transforms
