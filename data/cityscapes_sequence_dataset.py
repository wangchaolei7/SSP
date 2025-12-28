import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.video.video_dataset import compute_homo_cv


class CITYSCAPES_SEQ_CORRUPT:
    name = "CityscapesSequenceCorruptions"
    n_classes = 15
    fps = 17
    n_frames_vc = 8
    img_extension = ".png"
    label_extension = ".png"
    ignore_index = 255
    @staticmethod
    def convert_labels(mask):
        return mask
    classes = {
        0: "background",
        1: "class_0",
        2: "class_1",
        3: "class_2",
        4: "class_3",
        5: "class_4",
        6: "class_5",
        7: "class_6",
        8: "class_7",
        9: "class_8",
        10: "class_9",
        11: "class_10",
        12: "class_11",
        13: "class_12",
        14: "class_13",
        15: "class_14",
    }
    colors = {
        0: (0, 0, 0),
        1: (128, 64, 128),
        2: (244, 35, 232),
        3: (70, 70, 70),
        4: (102, 102, 156),
        5: (190, 153, 153),
        6: (153, 153, 153),
        7: (250, 170, 30),
        8: (220, 220, 0),
        9: (107, 142, 35),
        10: (152, 251, 152),
        11: (70, 130, 180),
        12: (220, 20, 60),
        13: (255, 0, 0),
        14: (0, 0, 142),
        15: (0, 60, 100),
    }


_IMAGE_SUFFIX = "_leftImg8bit"
_LABEL_SUFFIXES = (
    "_gtFine_label14TrainIds(TrainIds)",
    "_gtFine_label14TrainIds",
)


def _base_id_from_image(name: str) -> str:
    stem = os.path.splitext(name)[0]
    if stem.endswith(_IMAGE_SUFFIX):
        stem = stem[: -len(_IMAGE_SUFFIX)]
    return stem


def _base_id_from_label(name: str) -> str:
    stem = os.path.splitext(name)[0]
    for suffix in _LABEL_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return ""


def _canonical_label_name(base_id: str) -> str:
    return f"{base_id}{_LABEL_SUFFIXES[-1]}.png"


class CityscapesSequenceCorruptionsDataset(Dataset):
    def __init__(
        self,
        root_images: str,
        root_labels: str,
        corruption: str,
        DATASET: CITYSCAPES_SEQ_CORRUPT,
        data_cfg: dict,
        split: str = "val",
        img_transforms=None,
        segmentation_transforms=None,
        val_skip_frames: int = 1,
        log_stats: bool = False,
    ):
        super().__init__()
        self.root_images = root_images
        self.root_labels = root_labels
        self.corruption = corruption
        self.split = split
        self.n_classes = DATASET.n_classes
        self.ignore_index = DATASET.ignore_index
        self.img_extension = DATASET.img_extension
        self.label_extension = DATASET.label_extension
        self.crop_size = data_cfg["crop_size"]
        self.min_vid_len = data_cfg.get("min_vid_len", 0)
        self.val_skip_frames = val_skip_frames
        self.opencv_homos = data_cfg.get("opencv_homos", False)
        self.opencv_model_type = data_cfg.get("opencv_model_type", "sift")

        self.img_transforms = img_transforms
        self.segmentation_transforms = segmentation_transforms

        self.videos = []
        self.total_images = 0
        self.total_labels = 0
        self.total_matched = 0

        self._scan_dataset(log_stats)

    def _scan_dataset(self, log_stats: bool):
        corruption_root = os.path.join(self.root_images, self.corruption)
        if not os.path.isdir(corruption_root):
            raise FileNotFoundError(f"Cityscapes corruption root not found: {corruption_root}")

        for city in sorted(os.listdir(corruption_root)):
            city_dir = os.path.join(corruption_root, city)
            if not os.path.isdir(city_dir):
                continue
            for seq in sorted(os.listdir(city_dir)):
                img_dir = os.path.join(city_dir, seq)
                if not os.path.isdir(img_dir):
                    continue

                label_dir = os.path.join(self.root_labels, city, seq)
                if not os.path.isdir(label_dir):
                    continue

                image_files = [
                    f
                    for f in os.listdir(img_dir)
                    if f.endswith(self.img_extension) and _IMAGE_SUFFIX in f
                ]
                image_files.sort()
                self.total_images += len(image_files)

                label_files = [
                    f
                    for f in os.listdir(label_dir)
                    if f.endswith(self.label_extension) and "label14TrainIds" in f
                ]
                label_files.sort()
                self.total_labels += len(label_files)

                label_map: Dict[str, str] = {}
                for label_file in label_files:
                    base_id = _base_id_from_label(label_file)
                    if not base_id:
                        continue
                    logical_name = _canonical_label_name(base_id)
                    label_path = os.path.join(label_dir, label_file)
                    if logical_name not in label_map or "(TrainIds)" not in label_file:
                        label_map[logical_name] = label_path

                if not label_map:
                    continue

                matched = []
                for img_file in image_files:
                    base_id = _base_id_from_image(img_file)
                    logical_label = _canonical_label_name(base_id)
                    label_path = label_map.get(logical_label)
                    if label_path is None:
                        continue
                    matched.append(
                        (
                            img_file,
                            os.path.join(img_dir, img_file),
                            logical_label,
                            label_path,
                        )
                    )

                if len(matched) < self.min_vid_len + 1:
                    continue

                self.total_matched += len(matched)

                filtered = [
                    item for idx, item in enumerate(matched) if self.isinfered(idx)
                ]
                if not filtered:
                    continue

                frames_names = [item[0] for item in filtered]
                frame_paths = [item[1] for item in filtered]
                labels_names = [item[2] for item in filtered]
                label_paths = [item[3] for item in filtered]
                v_name = os.path.join(city, seq)

                self.videos.append(
                    {
                        "frames_names": frames_names,
                        "frame_paths": frame_paths,
                        "labels_names": labels_names,
                        "label_paths": label_paths,
                        "v_name": v_name,
                    }
                )

        if log_stats:
            print(
                "Cityscapes corruptions [{}]: images={}, labels={}, matched={}".format(
                    self.corruption, self.total_images, self.total_labels, self.total_matched
                )
            )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int):
        video = self.videos[index]
        frames_names = video["frames_names"]
        labels_names = video["labels_names"]
        v_name = video["v_name"]

        frames = [self.read_image(p) for p in video["frame_paths"]]
        labels = [self.read_mask(p) for p in video["label_paths"]]

        if self.opencv_homos:
            homos = [None] + [
                torch.tensor(
                    compute_homo_cv(frames[i - 1], frames[i], self.opencv_model_type),
                    dtype=torch.float32,
                ).unsqueeze(0)
                for i in range(1, len(frames))
            ]
        else:
            homos = [None for _ in frames]

        if self.img_transforms is not None:
            frames = [self.img_transforms(image=frame)["image"] for frame in frames]
        if self.segmentation_transforms is not None:
            labels = [
                self.segmentation_transforms(Image.fromarray(label)) for label in labels
            ]

        return frames_names, frames, labels_names, labels, v_name, homos

    def isinfered(self, idx_in_vid: int) -> bool:
        return idx_in_vid % self.val_skip_frames == 0 or idx_in_vid % self.val_skip_frames == 1

    def name_to_labelname(self, name: str) -> str:
        base_id = _base_id_from_image(name)
        return _canonical_label_name(base_id)

    def labelname_to_name(self, labelname: str) -> str:
        base_id = _base_id_from_label(labelname)
        return f"{base_id}{_IMAGE_SUFFIX}{self.img_extension}"

    def read_image(self, img_path: str) -> np.ndarray:
        image = Image.open(img_path)
        image = image.resize((self.crop_size[1], self.crop_size[0]), Image.BILINEAR)
        return np.array(image)

    def read_mask(self, mask_path: str) -> np.ndarray:
        label = Image.open(mask_path)
        label = label.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)
        return np.array(label)
