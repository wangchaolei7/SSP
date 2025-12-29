import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from data.utils.images_transforms import SoftLabelResize
from data.video.video_dataset import compute_homo_cv
from data.utils import resize_right, interp_methods
from utils.distributed import is_main_process


_IMAGE_DIR = "image_00"


def _normalize_extension(ext):
    if ext is None:
        return ""
    return ext if ext.startswith(".") else f".{ext}"


def _swap_split(path, split):
    if not path:
        return path
    needle = f"/{split}/"
    if needle in path:
        return path
    if "/train/" in path:
        return path.replace("/train/", f"/{split}/")
    if "/val/" in path:
        return path.replace("/val/", f"/{split}/")
    return path


def _is_frame_name(name, extension):
    stem, ext = os.path.splitext(name)
    return ext.lower() == extension.lower() and stem.isdigit() and len(stem) == 10


class Kitti360Base:
    def __init__(self, data_folder, DATASET, data_cfg, split):
        self.data_folder = data_folder
        self.split = split
        self.n_classes = data_cfg.get("num_classes", DATASET.n_classes)
        self.ignore_index = data_cfg.get("ignore_index", DATASET.ignore_index)
        self.label_extension = _normalize_extension(data_cfg.get("label_extension", DATASET.label_extension))
        self.img_extensions = self._get_img_extensions(data_cfg, DATASET)
        self.img_extension = self.img_extensions[0] if self.img_extensions else ""
        self.frame_folder = data_cfg.get("frame_folder", DATASET.frame_folder)
        self.mask_folder = data_cfg.get("mask_folder", DATASET.mask_folder)
        self.convert_labels = DATASET.convert_labels

        self.root_images, self.root_labels = self._resolve_roots(data_cfg)

        if not os.path.isdir(self.root_images):
            raise FileNotFoundError(f"KITTI-360 images root not found: {self.root_images}")
        if not os.path.isdir(self.root_labels):
            raise FileNotFoundError(f"KITTI-360 labels root not found: {self.root_labels}")

    def _get_img_extensions(self, data_cfg, DATASET):
        exts_cfg = data_cfg.get("img_extensions")
        if exts_cfg:
            exts = [_normalize_extension(ext).lower() for ext in exts_cfg]
        else:
            exts = [".jpg", ".jpeg", ".png"]
            preferred = data_cfg.get("img_extension", DATASET.img_extension)
            if preferred:
                preferred = _normalize_extension(preferred).lower()
                if preferred in exts:
                    exts.remove(preferred)
                exts.insert(0, preferred)
        return exts

    def _resolve_roots(self, data_cfg):
        root_images = data_cfg.get("root_images")
        root_labels = data_cfg.get("root_labels")
        if root_images and root_labels:
            if self.split != "train":
                split_images = data_cfg.get(f"root_images_{self.split}")
                split_labels = data_cfg.get(f"root_labels_{self.split}")
                if split_images and split_labels:
                    root_images = split_images
                    root_labels = split_labels
                else:
                    base_path = data_cfg.get("path")
                    candidate_images = os.path.join(base_path, self.split, "data_2d_raw") if base_path else None
                    candidate_labels = os.path.join(base_path, self.split, "data_2d_semantics") if base_path else None
                    if candidate_images and candidate_labels and os.path.isdir(candidate_images) and os.path.isdir(candidate_labels):
                        root_images = candidate_images
                        root_labels = candidate_labels
                    else:
                        root_images = _swap_split(root_images, self.split)
                        root_labels = _swap_split(root_labels, self.split)
        else:
            base_path = data_cfg.get("path") or self.data_folder
            if not base_path:
                raise ValueError("kitti360 requires data_cfg.path or data_cfg.root_images/root_labels")
            root_images = os.path.join(base_path, self.split, "data_2d_raw")
            root_labels = os.path.join(base_path, self.split, "data_2d_semantics")
        return root_images, root_labels

    def name_to_labelname(self, name):
        stem = os.path.splitext(name)[0]
        return f"{stem}{self.label_extension}"

    def labelname_to_name(self, labelname):
        stem = os.path.splitext(labelname)[0]
        return f"{stem}{self.img_extension}"

    def _frame_dir(self, v_name):
        return os.path.join(self.root_images, v_name, _IMAGE_DIR, self.frame_folder)

    def _mask_dir(self, v_name):
        return os.path.join(self.root_labels, v_name, _IMAGE_DIR, self.mask_folder)

    def _scan_record(self, v_name):
        frame_dir = self._frame_dir(v_name)
        label_dir = self._mask_dir(v_name)
        stats = {"total_gt": 0, "paired": 0, "skipped": 0}
        warnings = []

        if not os.path.isdir(label_dir):
            if is_main_process():
                warnings.append(f"KITTI-360 labels missing: {label_dir}")
            return [], set(), stats, warnings

        label_files = [
            f for f in os.listdir(label_dir)
            if _is_frame_name(f, self.label_extension)
        ]
        label_files.sort()
        stats["total_gt"] = len(label_files)

        if not os.path.isdir(frame_dir):
            stats["skipped"] = stats["total_gt"]
            if stats["skipped"] and is_main_process():
                warnings.append(
                    f"KITTI-360 images missing: {frame_dir} (skipped {stats['skipped']} labels)"
                )
            return [], set(), stats, warnings

        matched_frames = []
        matched_labels = []
        missing_images = []
        for label_name in label_files:
            img_name = self.labelname_to_name(label_name)
            if os.path.isfile(os.path.join(frame_dir, img_name)):
                matched_frames.append(img_name)
                matched_labels.append(label_name)
            else:
                missing_images.append(label_name)

        stats["paired"] = len(matched_frames)
        stats["skipped"] = stats["total_gt"] - stats["paired"]
        if missing_images and is_main_process():
            warnings.append(
                f"KITTI-360 pairing issue in {self.split}/{v_name}: "
                f"{stats['skipped']} labels without images (e.g. {missing_images[:5]})"
            )

        return matched_frames, set(matched_labels), stats, warnings


class Kitti360ImageDataset(Dataset, Kitti360Base):
    def __init__(
        self,
        data_folder,
        video_indices,
        DATASET,
        data_cfg,
        split="train",
        training=True,
        joint_transforms=None,
        img_transforms=None,
        segmentation_transforms=None,
    ):
        Dataset.__init__(self)
        Kitti360Base.__init__(self, data_folder, DATASET, data_cfg, split)
        self.crop_size = data_cfg["crop_size"]
        self.labeled_frames_per_vid = data_cfg.get("labeled_frames_per_vid", None)
        self.val_skip_frames = data_cfg.get("val_skip_frames", 1)
        self.val_max_samples = data_cfg.get("val_max_samples")
        self.min_vid_len = data_cfg.get("min_vid_len", 0)
        self.soft_labels = data_cfg.get("soft_labels", False)
        self.training = training

        self.samples = []
        self.frames_by_video = {}
        self.labels_by_video = {}
        self._scan_stats = {"total_gt": 0, "paired": 0, "skipped": 0}
        self._scan_warnings = []

        for v_name in video_indices:
            frames, label_set, stats, warnings = self._scan_record(v_name)
            for key in self._scan_stats:
                self._scan_stats[key] += stats[key]
            self._scan_warnings.extend(warnings)

            if len(frames) < self.min_vid_len + 1:
                continue

            self.frames_by_video[v_name] = frames
            self.labels_by_video[v_name] = label_set

            for (i, f) in enumerate(frames):
                if self.islabeled(f, v_name, i, len(frames)):
                    self.samples.append((f, v_name))

        self._val_samples_limited = False
        self._limit_val_samples()

        self.joint_transforms = joint_transforms
        self.img_transforms = img_transforms
        self.segmentation_transforms = segmentation_transforms

        self._log_scan_stats()

    def _limit_val_samples(self):
        if self.training or self.val_max_samples is None:
            return
        if len(self.samples) <= self.val_max_samples:
            return
        indices = np.linspace(0, len(self.samples) - 1, self.val_max_samples, dtype=int)
        self.samples = [self.samples[i] for i in indices]
        self._val_samples_limited = True

    def _log_scan_stats(self):
        if not is_main_process():
            return
        print(
            "KITTI-360 {} scan: total_gt={}, paired={}, skipped={}".format(
                self.split,
                self._scan_stats["total_gt"],
                self._scan_stats["paired"],
                self._scan_stats["skipped"],
            )
        )
        if self._val_samples_limited:
            print(f"KITTI-360 {self.split} samples limited to {len(self.samples)} (val_max_samples)")
        if self._scan_warnings:
            for warning in self._scan_warnings[:5]:
                print(warning)
            if len(self._scan_warnings) > 5:
                print("KITTI-360 pairing warnings truncated")

    def __getitem__(self, index):
        f, v_name = self.samples[index]
        label_name = self.name_to_labelname(f)

        frame = self.read_image(f, v_name)
        label = self.read_mask(label_name, v_name)

        if self.joint_transforms is not None:
            transformed = self.joint_transforms(image=frame, mask=label)
            frame, label = transformed["image"], transformed["mask"]
        if self.img_transforms is not None:
            frame = self.img_transforms(image=frame)["image"]
        if self.segmentation_transforms is not None:
            label = self.segmentation_transforms(label)

        return frame, label

    def __len__(self):
        return len(self.samples)

    def islabeled(self, f, v_name, idx_in_vid, len_vid):
        labels = self.labels_by_video.get(v_name, set())
        if not self.training:
            if idx_in_vid % self.val_skip_frames == 0:
                return self.name_to_labelname(f) in labels
            return False
        if self.labeled_frames_per_vid is None:
            return self.name_to_labelname(f) in labels
        n_labeled_frames = min(self.labeled_frames_per_vid, len_vid)
        labels_idx = np.linspace(0, len_vid - 1, n_labeled_frames, dtype=int)
        return idx_in_vid in labels_idx

    def read_image(self, img, v_name):
        i = Image.open(os.path.join(self._frame_dir(v_name), img))
        i = i.resize((self.crop_size[1], self.crop_size[0]), Image.BILINEAR)
        return np.array(i)

    def read_mask(self, mask, v_name):
        l = Image.open(os.path.join(self._mask_dir(v_name), mask))
        if self.soft_labels:
            l = SoftLabelResize(
                self.n_classes, self.crop_size, self.convert_labels, self.ignore_index, "bilinear"
            )(l)
        else:
            l = l.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)
            l = np.array(l)
        return l


class Kitti360ImageInferenceDataset(Dataset, Kitti360Base):
    def __init__(
        self,
        data_folder,
        video_indices,
        DATASET,
        data_cfg,
        split="val",
        img_transforms=None,
        segmentation_transforms=None,
        val_skip_frames=1,
    ):
        Dataset.__init__(self)
        Kitti360Base.__init__(self, data_folder, DATASET, data_cfg, split)
        self.crop_size = data_cfg["crop_size"]
        self.min_vid_len = data_cfg.get("min_vid_len", 0)
        self.val_skip_frames = val_skip_frames
        self.opencv_homos = data_cfg.get("opencv_homos", False)
        self.opencv_model_type = data_cfg.get("opencv_model_type", "sift")

        self.videos = []
        self.labels_by_video = {}
        self._scan_stats = {"total_gt": 0, "paired": 0, "skipped": 0}
        self._scan_warnings = []

        for v_name in video_indices:
            frames, label_set, stats, warnings = self._scan_record(v_name)
            for key in self._scan_stats:
                self._scan_stats[key] += stats[key]
            self._scan_warnings.extend(warnings)

            if len(frames) < self.min_vid_len + 1:
                continue
            self.videos.append((frames, v_name))
            self.labels_by_video[v_name] = label_set

        self.img_transforms = img_transforms
        self.segmentation_transforms = segmentation_transforms

        self._log_scan_stats()

    def _log_scan_stats(self):
        if not is_main_process():
            return
        print(
            "KITTI-360 {} scan: total_gt={}, paired={}, skipped={}".format(
                self.split,
                self._scan_stats["total_gt"],
                self._scan_stats["paired"],
                self._scan_stats["skipped"],
            )
        )
        if self._scan_warnings:
            for warning in self._scan_warnings[:5]:
                print(warning)
            if len(self._scan_warnings) > 5:
                print("KITTI-360 pairing warnings truncated")

    def __getitem__(self, index):
        frames_names, v_name = self.videos[index]
        frames_names = [f for (i, f) in enumerate(frames_names) if self.isinfered(i)]
        labels_names = [
            self.name_to_labelname(f)
            for (i, f) in enumerate(frames_names)
            if self.islabeled(f, v_name)
        ]

        frames = [self.read_image(f, v_name) for f in frames_names]
        labels = [self.read_mask(l, v_name) for l in labels_names]

        if self.opencv_homos:
            homos = [
                None
            ] + [
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

    def islabeled(self, f, v_name):
        labels = self.labels_by_video.get(v_name, set())
        return self.name_to_labelname(f) in labels

    def read_image(self, img, v_name):
        i = Image.open(os.path.join(self._frame_dir(v_name), img))
        i = i.resize((self.crop_size[1], self.crop_size[0]), Image.BILINEAR)
        return np.array(i)

    def read_mask(self, mask, v_name):
        l = Image.open(os.path.join(self._mask_dir(v_name), mask))
        l = l.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)
        return np.array(l)


class Kitti360ImageLogitsDataset(Kitti360ImageDataset):
    def __init__(
        self,
        data_folder,
        video_indices,
        DATASET,
        data_cfg,
        split="train",
        training=True,
        joint_transforms=None,
        img_transforms=None,
        segmentation_transforms=None,
    ):
        data_cfg = dict(data_cfg)
        data_cfg["strict_pairs"] = False
        self.logit_folder = data_cfg["logits_folder"]
        self.train_skip_frames = data_cfg.get("train_skip_frames", None)
        super().__init__(
            data_folder,
            video_indices,
            DATASET,
            data_cfg,
            split=split,
            training=training,
            joint_transforms=joint_transforms,
            img_transforms=img_transforms,
            segmentation_transforms=segmentation_transforms,
        )
        self.label_extension = ".npz"
        self.mask_folder = ""

    def islabeled(self, f, v_name, idx_in_vid, len_vid):
        if not self.training:
            return idx_in_vid % self.val_skip_frames == 0
        if self.train_skip_frames is not None:
            labels_idx = np.arange(1, len_vid, step=self.train_skip_frames)
            return idx_in_vid in labels_idx
        if self.labeled_frames_per_vid is None:
            return True
        n_labeled_frames = min(self.labeled_frames_per_vid, len_vid)
        labels_idx = np.arange(
            1, n_labeled_frames * (len_vid // n_labeled_frames) + 1, step=len_vid // n_labeled_frames
        )
        return idx_in_vid in labels_idx

    def read_mask(self, mask, v_name):
        l = np.load(os.path.join(self.logit_folder, v_name, mask))["arr_0"].transpose(1, 2, 0)
        l = resize_right.resize(
            l,
            out_shape=(self.crop_size[0], self.crop_size[1]),
            interp_method=interp_methods.linear,
            support_sz=2,
            antialiasing=True,
        )
        return l


class Kitti360VideoDataset(Dataset, Kitti360Base):
    def __init__(
        self,
        data_folder,
        video_indices,
        DATASET,
        data_cfg,
        split="train",
        training=True,
        joint_transforms=None,
        img_transforms=None,
        segmentation_transforms=None,
    ):
        Dataset.__init__(self)
        Kitti360Base.__init__(self, data_folder, DATASET, data_cfg, split)
        self.crop_size = data_cfg["crop_size"]
        self.adjacent_frames = data_cfg["adjacent_frames"]
        self.labeled_frames_per_vid = data_cfg.get("labeled_frames_per_vid", None)
        self.val_skip_frames = data_cfg.get("val_skip_frames", 1)
        self.val_max_samples = data_cfg.get("val_max_samples")
        self.min_vid_len = data_cfg.get("min_vid_len", 0)
        self.soft_labels = data_cfg.get("soft_labels", False)
        self.opencv_homos = data_cfg.get("opencv_homos", False)
        self.opencv_model_type = data_cfg.get("opencv_model_type", "sift")
        self.training = training

        self.samples = []
        self.videos = []
        self.video_names = []
        self.labels_by_video = {}
        self._scan_stats = {"total_gt": 0, "paired": 0, "skipped": 0}
        self._scan_warnings = []

        v_idx = 0
        for v_name in video_indices:
            frames, label_set, stats, warnings = self._scan_record(v_name)
            for key in self._scan_stats:
                self._scan_stats[key] += stats[key]
            self._scan_warnings.extend(warnings)

            min_vid_len = (
                max(
                    self.min_vid_len,
                    max(0, max(self.adjacent_frames)) - min(0, min(self.adjacent_frames)),
                )
                if len(self.adjacent_frames) > 0
                else self.min_vid_len
            )
            if len(frames) < min_vid_len + 1:
                continue

            frames_iter = (
                frames[:-max(0, max(self.adjacent_frames))]
                if max(0, max(self.adjacent_frames)) > 0
                else frames
            )
            for (i, f) in enumerate(frames_iter, -min(0, min(self.adjacent_frames))):
                if self.islabeled(f, v_name, i, len(frames_iter)):
                    adj_frames = [
                        frames[i - (-min(0, min(self.adjacent_frames))) + k] for k in self.adjacent_frames
                    ]
                    self.samples.append((f, adj_frames, (v_idx, i)))

            self.videos.append(frames)
            self.video_names.append(v_name)
            self.labels_by_video[v_name] = label_set
            v_idx += 1

        self._val_samples_limited = False
        self._limit_val_samples()

        self.joint_transforms = joint_transforms
        self.img_transforms = img_transforms
        self.segmentation_transforms = segmentation_transforms

        self._log_scan_stats()

    def _limit_val_samples(self):
        if self.training or self.val_max_samples is None:
            return
        if len(self.samples) <= self.val_max_samples:
            return
        indices = np.linspace(0, len(self.samples) - 1, self.val_max_samples, dtype=int)
        self.samples = [self.samples[i] for i in indices]
        self._val_samples_limited = True

    def _log_scan_stats(self):
        if not is_main_process():
            return
        print(
            "KITTI-360 {} scan: total_gt={}, paired={}, skipped={}".format(
                self.split,
                self._scan_stats["total_gt"],
                self._scan_stats["paired"],
                self._scan_stats["skipped"],
            )
        )
        if self._val_samples_limited:
            print(f"KITTI-360 {self.split} samples limited to {len(self.samples)} (val_max_samples)")
        if self._scan_warnings:
            for warning in self._scan_warnings[:5]:
                print(warning)
            if len(self._scan_warnings) > 5:
                print("KITTI-360 pairing warnings truncated")

    def __getitem__(self, index):
        f, adj_frames, (v_idx, i) = self.samples[index]
        label_name = self.name_to_labelname(f)

        v_name = self.video_names[v_idx]
        frame = self.read_image(f, v_name)
        adj_frames = [self.read_image(adj_f, v_name) for adj_f in adj_frames]
        label = self.read_mask(label_name, v_name)

        if self.joint_transforms is not None:
            frame, adj_frames, _, label = self.joint_transforms(frame, adj_frames, None, label)

        if self.opencv_homos:
            homo = compute_homo_cv(adj_frames[0], frame, self.opencv_model_type)
            homo = torch.tensor(homo, dtype=torch.float32).unsqueeze(0)
        else:
            homo = torch.tensor([0])

        if self.img_transforms is not None:
            frame, adj_frames, _ = self.img_transforms(frame, adj_frames, None)
        if self.segmentation_transforms is not None:
            label = self.segmentation_transforms(label)

        return frame, adj_frames, label, homo

    def __len__(self):
        return len(self.samples)

    def islabeled(self, f, v_name, idx_in_vid, len_vid):
        labels = self.labels_by_video.get(v_name, set())
        if not self.training:
            if idx_in_vid % self.val_skip_frames == 0:
                return self.name_to_labelname(f) in labels
            return False
        if self.labeled_frames_per_vid is None:
            return self.name_to_labelname(f) in labels
        n_labeled_frames = min(self.labeled_frames_per_vid, len_vid)
        labels_idx = np.linspace(0, len_vid - 1, n_labeled_frames, dtype=int)
        return idx_in_vid in labels_idx

    def read_image(self, img, v_name):
        i = Image.open(os.path.join(self._frame_dir(v_name), img))
        i = i.resize((self.crop_size[1], self.crop_size[0]), Image.BILINEAR)
        return np.array(i)

    def read_mask(self, mask, v_name):
        l = Image.open(os.path.join(self._mask_dir(v_name), mask))
        if self.soft_labels:
            l = SoftLabelResize(
                self.n_classes, self.crop_size, self.convert_labels, self.ignore_index, "bilinear"
            )(l)
        else:
            l = l.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)
        return l


class Kitti360VideoLogitsDataset(Kitti360VideoDataset):
    def __init__(
        self,
        data_folder,
        video_indices,
        DATASET,
        data_cfg,
        split="train",
        training=True,
        joint_transforms=None,
        img_transforms=None,
        segmentation_transforms=None,
    ):
        data_cfg = dict(data_cfg)
        data_cfg["strict_pairs"] = False
        self.logit_folder = data_cfg["logits_folder"]
        self.train_skip_frames = data_cfg.get("train_skip_frames", None)
        super().__init__(
            data_folder,
            video_indices,
            DATASET,
            data_cfg,
            split=split,
            training=training,
            joint_transforms=joint_transforms,
            img_transforms=img_transforms,
            segmentation_transforms=segmentation_transforms,
        )
        self.label_extension = ".npz"
        self.mask_folder = ""

    def islabeled(self, f, v_name, idx_in_vid, len_vid):
        if not self.training:
            return idx_in_vid % self.val_skip_frames == 0
        if self.train_skip_frames is not None:
            labels_idx = np.arange(1, len_vid, step=self.train_skip_frames)
            return idx_in_vid in labels_idx
        if self.labeled_frames_per_vid is None:
            return True
        n_labeled_frames = min(self.labeled_frames_per_vid, len_vid)
        labels_idx = np.arange(n_labeled_frames * (len_vid // n_labeled_frames), step=len_vid // n_labeled_frames)
        return idx_in_vid in labels_idx

    def _mask_dir(self, v_name):
        return os.path.join(self.logit_folder, v_name)

    def read_mask(self, mask, v_name):
        l = np.load(os.path.join(self.logit_folder, v_name, mask))["arr_0"].transpose(1, 2, 0)
        l = resize_right.resize(
            l,
            out_shape=(self.crop_size[0], self.crop_size[1]),
            interp_method=interp_methods.linear,
            support_sz=2,
            antialiasing=True,
        )
        return l
