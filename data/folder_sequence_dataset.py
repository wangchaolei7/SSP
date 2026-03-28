import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.video.video_dataset import compute_homo_cv
from utils.distributed import is_main_process


COMMON_15_CLASSES = {
    0: "background",
    1: "road",
    2: "sidewalk",
    3: "building",
    4: "wall",
    5: "fence",
    6: "pole",
    7: "traffic light",
    8: "traffic sign",
    9: "vegetation",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "Truck_Bus",
}

COMMON_15_COLORS = {
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
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 60, 100),
}

FOLDER_SEQUENCE_PRESETS = {
    "folder_sequence": {
        "dataset": "folder_sequence",
        "num_classes": 15,
        "ignore_index": 255,
        "fps": 30,
        "frame_folder": "",
        "mask_folder": "",
        "img_extension": ".png",
        "label_extension": ".png",
        "label_suffix": "",
    },
    "uavid_crossdomain": {
        "dataset": "uavid_crossdomain",
        "path": "/data1/wangcl/dataset/open_video_DGSS/uavid_val",
        "root_images": "/data1/wangcl/dataset/open_video_DGSS/uavid_val",
        "root_labels": "/data1/wangcl/dataset/open_video_DGSS/uavid_val",
        "frame_folder": "Images",
        "mask_folder": "Labels_classes15",
        "img_extension": ".png",
        "label_extension": ".png",
        "label_suffix": "",
        "num_classes": 15,
        "ignore_index": 255,
        "fps": 20,
    },
    "vspw_crossdomain": {
        "dataset": "vspw_crossdomain",
        "path": "/data1/wangcl/dataset/open_video_DGSS/VSPW_val",
        "root_images": "/data1/wangcl/dataset/open_video_DGSS/VSPW_val/ColorImage",
        "root_labels": "/data1/wangcl/dataset/open_video_DGSS/VSPW_val/Label_classes15",
        "frame_folder": "",
        "mask_folder": "",
        "img_extension": ".jpg",
        "label_extension": ".png",
        "label_suffix": "",
        "num_classes": 15,
        "ignore_index": 255,
        "fps": 30,
    },
}


def _normalize_extension(ext: Optional[str]) -> str:
    if not ext:
        return ""
    return ext if ext.startswith(".") else f".{ext}"


def _parse_sequence_filter(value) -> Optional[Set[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple, set)):
        raw_items = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                raw_items.extend([part.strip() for part in item.split(",") if part.strip()])
            else:
                raw_items.append(str(item))
    else:
        raw_items = [str(value)]
    normalized = {os.path.basename(item.replace("\\", "/").strip()) for item in raw_items if item}
    return normalized or None


def _convert_common_15_labels(label, n_classes, ignore_index):
    label = np.asarray(label)
    label = label.copy()
    invalid = (label < 0) | (label >= n_classes)
    label[invalid] = ignore_index
    return label


def is_folder_sequence_dataset(name: Optional[str]) -> bool:
    return (name or "").lower() in FOLDER_SEQUENCE_PRESETS


def folder_sequence_defaults(name: Optional[str]) -> Dict[str, object]:
    return dict(FOLDER_SEQUENCE_PRESETS.get((name or "").lower(), {}))


def build_folder_sequence_dataset_def(data_cfg):
    dataset_name = data_cfg.get("dataset", "folder_sequence")
    preset = folder_sequence_defaults(dataset_name)
    merged = dict(preset)
    merged.update(data_cfg)

    class FolderSequenceDatasetDef:
        name = str(merged.get("dataset", dataset_name))
        n_classes = int(merged.get("num_classes", 15))
        fps = int(merged.get("fps", 30))
        img_extension = _normalize_extension(merged.get("img_extension", ".png"))
        label_extension = _normalize_extension(merged.get("label_extension", ".png"))
        frame_folder = merged.get("frame_folder", "") or ""
        mask_folder = merged.get("mask_folder", "") or ""
        label_suffix = merged.get("label_suffix", "") or ""
        ignore_index = int(merged.get("ignore_index", 255))
        classes = COMMON_15_CLASSES
        colors = COMMON_15_COLORS

        @staticmethod
        def convert_labels(label):
            return _convert_common_15_labels(
                label,
                FolderSequenceDatasetDef.n_classes,
                FolderSequenceDatasetDef.ignore_index,
            )

    return FolderSequenceDatasetDef


class FolderSequenceImageInferenceDataset(Dataset):
    def __init__(
        self,
        DATASET,
        data_cfg,
        img_transforms=None,
        segmentation_transforms=None,
        val_skip_frames=1,
        log_stats=False,
    ):
        super().__init__()
        self.n_classes = DATASET.n_classes
        self.ignore_index = DATASET.ignore_index
        self.img_extensions = self._get_img_extensions(data_cfg, DATASET)
        self.img_extension = self.img_extensions[0] if self.img_extensions else ""
        self.label_extension = _normalize_extension(
            data_cfg.get("label_extension", DATASET.label_extension)
        )
        self.frame_folder = data_cfg.get("frame_folder", getattr(DATASET, "frame_folder", "")) or ""
        self.mask_folder = data_cfg.get("mask_folder", getattr(DATASET, "mask_folder", "")) or ""
        self.label_suffix = data_cfg.get("label_suffix", getattr(DATASET, "label_suffix", "")) or ""
        self.crop_size = data_cfg["crop_size"]
        self.min_vid_len = data_cfg.get("min_vid_len", 0)
        self.val_skip_frames = val_skip_frames
        self.opencv_homos = data_cfg.get("opencv_homos", False)
        self.opencv_model_type = data_cfg.get("opencv_model_type", "sift")
        self.strict_pairs = data_cfg.get("strict_pairs", data_cfg.get("strict_pairing", False))
        self.sequence_filter = _parse_sequence_filter(
            data_cfg.get("sequence_filter", data_cfg.get("sequence_filters"))
        )
        self.root_images = os.path.abspath(os.path.expanduser(data_cfg["root_images"]))
        self.root_labels = os.path.abspath(os.path.expanduser(data_cfg["root_labels"]))

        if not os.path.isdir(self.root_images):
            raise FileNotFoundError(f"Folder-sequence images root not found: {self.root_images}")
        if not os.path.isdir(self.root_labels):
            raise FileNotFoundError(f"Folder-sequence labels root not found: {self.root_labels}")

        self.videos = []
        self.labels_by_video: Dict[str, Dict[str, str]] = {}
        self._scan_stats = {"videos": 0, "images": 0, "labels": 0, "paired": 0}
        self._scan_warnings: List[str] = []

        self.img_transforms = img_transforms
        self.segmentation_transforms = segmentation_transforms

        self._scan_dataset()
        if log_stats and is_main_process():
            print(
                "Folder-sequence {} scan: videos={}, images={}, labels={}, paired={}".format(
                    data_cfg.get("dataset", "folder_sequence"),
                    self._scan_stats["videos"],
                    self._scan_stats["images"],
                    self._scan_stats["labels"],
                    self._scan_stats["paired"],
                )
            )
            if self._scan_warnings:
                for warning in self._scan_warnings[:10]:
                    print(warning)
                if len(self._scan_warnings) > 10:
                    print("Folder-sequence pairing warnings truncated")

    def _get_img_extensions(self, data_cfg, DATASET):
        exts_cfg = data_cfg.get("img_extensions")
        if exts_cfg:
            return [_normalize_extension(ext).lower() for ext in exts_cfg if ext]
        default_exts = [".jpg", ".jpeg", ".png"]
        preferred = _normalize_extension(data_cfg.get("img_extension", DATASET.img_extension)).lower()
        if preferred:
            if preferred in default_exts:
                default_exts.remove(preferred)
            default_exts.insert(0, preferred)
        return default_exts

    def _frame_dir(self, v_name):
        parts = [self.root_images, v_name]
        if self.frame_folder:
            parts.append(self.frame_folder)
        return os.path.join(*parts)

    def _mask_dir(self, v_name):
        parts = [self.root_labels, v_name]
        if self.mask_folder:
            parts.append(self.mask_folder)
        return os.path.join(*parts)

    def _label_suffix_ext(self, extension=None):
        extension = _normalize_extension(extension or self.label_extension)
        return f"{self.label_suffix}{extension}" if self.label_suffix else extension

    def _label_stem(self, label_name):
        stem = os.path.splitext(label_name)[0]
        if self.label_suffix and stem.endswith(self.label_suffix):
            stem = stem[: -len(self.label_suffix)]
        return stem

    def name_to_labelname(self, name, label_extension=None):
        stem = os.path.splitext(name)[0]
        return f"{stem}{self._label_suffix_ext(label_extension or self.label_extension)}"

    def _scan_record(self, v_name) -> Tuple[List[str], Dict[str, str], Dict[str, int], List[str]]:
        frame_dir = self._frame_dir(v_name)
        label_dir = self._mask_dir(v_name)
        stats = {"images": 0, "labels": 0, "paired": 0}
        warnings = []

        if not os.path.isdir(frame_dir):
            warnings.append(f"Folder-sequence images missing: {frame_dir}")
            return [], {}, stats, warnings
        if not os.path.isdir(label_dir):
            warnings.append(f"Folder-sequence labels missing: {label_dir}")
            return [], {}, stats, warnings

        ext_priority = {ext: idx for (idx, ext) in enumerate(self.img_extensions)}
        image_map = {}
        for file_name in os.listdir(frame_dir):
            stem, ext = os.path.splitext(file_name)
            ext = ext.lower()
            if ext not in ext_priority:
                continue
            if stem not in image_map or ext_priority[ext] < ext_priority[image_map[stem][1]]:
                image_map[stem] = (file_name, ext)
        image_map = {stem: value[0] for (stem, value) in image_map.items()}

        label_suffix_ext = self._label_suffix_ext(self.label_extension).lower()
        label_map = {}
        for file_name in os.listdir(label_dir):
            if not file_name.lower().endswith(label_suffix_ext):
                continue
            logical_name = self.name_to_labelname(self._label_stem(file_name), self.label_extension)
            label_map[logical_name] = file_name

        stats["images"] = len(image_map)
        stats["labels"] = len(label_map)

        image_stems = set(image_map.keys())
        label_stems = {self._label_stem(name) for name in label_map}
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        if missing_labels or missing_images:
            parts = []
            if missing_labels:
                parts.append(
                    f"{len(missing_labels)} images without labels (e.g. {missing_labels[:5]})"
                )
            if missing_images:
                parts.append(
                    f"{len(missing_images)} labels without images (e.g. {missing_images[:5]})"
                )
            msg = f"Folder-sequence pairing issue in {v_name}: " + "; ".join(parts)
            if self.strict_pairs:
                raise FileNotFoundError(msg)
            warnings.append(msg)

        frames_all = [image_map[stem] for stem in sorted(image_stems)]
        paired_labels = {}
        for frame_name in frames_all:
            logical_label = self.name_to_labelname(frame_name, self.label_extension)
            actual_label = label_map.get(logical_label)
            if actual_label is not None:
                paired_labels[logical_label] = actual_label
        stats["paired"] = len(paired_labels)
        return frames_all, paired_labels, stats, warnings

    def _scan_dataset(self):
        video_names = [
            name
            for name in sorted(os.listdir(self.root_images))
            if os.path.isdir(os.path.join(self.root_images, name))
        ]
        for v_name in video_names:
            if self.sequence_filter and v_name not in self.sequence_filter:
                continue
            frames_all, label_map, stats, warnings = self._scan_record(v_name)
            self._scan_stats["images"] += stats["images"]
            self._scan_stats["labels"] += stats["labels"]
            self._scan_stats["paired"] += stats["paired"]
            self._scan_warnings.extend(warnings)

            if len(frames_all) < self.min_vid_len + 1:
                continue

            self.videos.append((frames_all, v_name))
            self.labels_by_video[v_name] = label_map
            self._scan_stats["videos"] += 1

    def __getitem__(self, index):
        frames_all, v_name = self.videos[index]
        frames_names = [name for (i, name) in enumerate(frames_all) if self.isinfered(i)]
        labels_names = [
            self.name_to_labelname(name, self.label_extension)
            for name in frames_names
            if self.islabeled(name, v_name)
        ]

        frames = [self.read_image(name, v_name) for name in frames_names]
        labels = [self.read_mask(name, v_name) for name in labels_names]

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
            labels = [self.segmentation_transforms(Image.fromarray(label)) for label in labels]

        return frames_names, frames, labels_names, labels, v_name, homos

    def __len__(self):
        return len(self.videos)

    def isinfered(self, idx_in_vid):
        return idx_in_vid % self.val_skip_frames == 0 or idx_in_vid % self.val_skip_frames == 1

    def islabeled(self, frame_name, v_name):
        label_name = self.name_to_labelname(frame_name, self.label_extension)
        return label_name in self.labels_by_video.get(v_name, {})

    def read_image(self, image_name, v_name):
        image = Image.open(os.path.join(self._frame_dir(v_name), image_name))
        image = image.resize((self.crop_size[1], self.crop_size[0]), Image.BILINEAR)
        return np.array(image)

    def read_mask(self, label_name, v_name):
        actual_label = self.labels_by_video[v_name][label_name]
        label = Image.open(os.path.join(self._mask_dir(v_name), actual_label))
        label = label.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)
        return np.array(label)
