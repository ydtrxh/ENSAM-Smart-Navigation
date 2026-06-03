"""
Step 2 — dataset.py
===================
Dataset, Sampler, and Transforms for the retraining pipeline.
Class list loaded exclusively from data/buildings.json — never hardcoded.

ClassMismatchError is raised as a hard error (not a warning) if any subfolder
in data/train/ is not present in buildings.json.
"""

import json
import logging
import os
import random
from collections import defaultdict
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ClassMismatchError(Exception):
    """
    Raised when a subfolder in data/train/ is not listed in buildings.json.
    This is a contract violation — data/train/ must never contain unknown classes.
    """


def load_buildings(data_dir: str) -> list:
    path = os.path.join(data_dir, "buildings.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"buildings.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_train_transforms() -> T.Compose:
    """
    On-the-fly augmentation applied to each image at training time.

    IMPORTANT: RandomErasing operates on TENSORS, not PIL Images.
    It MUST appear AFTER ToTensor() and Normalize().
    Placing it before ToTensor() raises TypeError at runtime.
    """
    return T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.RandomPerspective(distortion_scale=0.3),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # RandomErasing operates on tensors, not PIL Images.
        # Must appear AFTER ToTensor() and Normalize().
        # Placing it before raises TypeError. Do NOT move it.
        T.RandomErasing(p=0.2),
    ])


def get_val_transforms() -> T.Compose:
    """
    Deterministic transform for validation / inference / gallery building.
    No randomness — ensures reproducible embeddings.
    """
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class BuildingsDataset(Dataset):
    """
    Training dataset.
    - Loads class list exclusively from data/buildings.json at instantiation.
    - Scans data/train/<label>/ for each label.
    - Raises ClassMismatchError if any subfolder in data/train/ is not in buildings.json.

    Attributes:
        classes (list[str]): ordered list of class labels (from buildings.json)
        class_to_idx (dict[str, int]): label → integer index
        node_ids (list[str]): parallel list of Neo4j node_ids
    """

    def __init__(self, data_dir: str, transform: Optional[T.Compose] = None) -> None:
        self.data_dir  = data_dir
        self.transform = transform

        buildings          = load_buildings(data_dir)
        self.classes       = [b["label"]   for b in buildings]
        self.node_ids      = [b["node_id"] for b in buildings]
        self.class_to_idx  = {label: idx for idx, label in enumerate(self.classes)}
        self.label_to_node = {b["label"]: b["node_id"] for b in buildings}

        train_dir = os.path.join(data_dir, "train")

        # Contract check: no rogue subfolders
        if os.path.isdir(train_dir):
            label_set = set(self.classes)
            for folder in os.listdir(train_dir):
                if os.path.isdir(os.path.join(train_dir, folder)) and folder not in label_set:
                    raise ClassMismatchError(
                        f"data/train/{folder!r} is not in buildings.json. "
                        "Run sync_check.py to diagnose."
                    )

        self.image_paths: list[str] = []
        self.targets:     list[int] = []

        for label in self.classes:
            class_dir = os.path.join(train_dir, label)
            if not os.path.isdir(class_dir):
                logger.warning(f"'{label}': no data/train/{label}/ folder — class will have 0 images")
                continue
            for fname in os.listdir(class_dir):
                if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.targets.append(self.class_to_idx[label])

        logger.info(
            f"BuildingsDataset: {len(self.classes)} classes, "
            f"{len(self.image_paths)} images loaded from {train_dir}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path  = self.image_paths[idx]
        label = self.targets[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class GalleryDataset(Dataset):
    """
    Gallery / validation dataset.
    Loads images from a folder structure (gallery/ or val/).
    Class list validated against buildings.json.
    Returns (image_tensor, class_label_str).
    """

    def __init__(self, folder: str, data_dir: str, transform: Optional[T.Compose] = None) -> None:
        if transform is None:
            transform = get_val_transforms()
        self.transform = transform

        buildings  = load_buildings(data_dir)
        label_set  = {b["label"] for b in buildings}

        self.image_paths:  list[str] = []
        self.class_labels: list[str] = []

        if not os.path.isdir(folder):
            logger.warning(f"GalleryDataset: folder does not exist: {folder}")
            return

        for class_name in sorted(os.listdir(folder)):
            class_dir = os.path.join(folder, class_name)
            if not os.path.isdir(class_dir):
                continue
            if class_name not in label_set:
                raise ClassMismatchError(
                    f"{folder}/{class_name!r} is not in buildings.json"
                )
            for fname in os.listdir(class_dir):
                if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.class_labels.append(class_name)

        logger.debug(f"GalleryDataset: {len(self.image_paths)} images from {folder}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path  = self.image_paths[idx]
        label = self.class_labels[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label


class PKSampler(Sampler):
    """
    Batch sampler: each batch contains exactly P classes × K images per class.

    - P = min(batch_p, num_classes) — never hardcoded.
    - If a class has fewer than K images, sample with replacement for that class only.
      A WARNING is logged naming the affected class (once per instantiation).
    - Each epoch shuffles class order and within-class image order.
    """

    def __init__(self, targets: list, p: int, k: int) -> None:
        self.k = k
        self.class_to_indices: dict = defaultdict(list)
        for idx, cid in enumerate(targets):
            self.class_to_indices[cid].append(idx)

        self.classes   = list(self.class_to_indices.keys())
        self.p         = min(p, len(self.classes))
        self.batch_size = self.p * self.k

        if self.p < p:
            logger.warning(
                f"PKSampler: requested P={p} classes per batch but only "
                f"{len(self.classes)} classes available — using P={self.p}"
            )

        # Warn about classes with fewer than K images (once)
        for cid, idxs in self.class_to_indices.items():
            if len(idxs) < self.k:
                logger.warning(
                    f"PKSampler: class {cid} has only {len(idxs)} images "
                    f"(< K={self.k}) — will oversample with replacement"
                )

    def __iter__(self):
        class_to_indices = {c: list(idxs) for c, idxs in self.class_to_indices.items()}
        for idxs in class_to_indices.values():
            random.shuffle(idxs)

        available = list(self.classes)
        random.shuffle(available)

        while len(available) >= self.p:
            sampled_classes = random.sample(available, self.p)
            batch = []

            for c in sampled_classes:
                pool = class_to_indices[c]
                if len(pool) >= self.k:
                    selected = pool[:self.k]
                    class_to_indices[c] = pool[self.k:]
                else:
                    # Oversample with replacement from the full index set
                    selected = pool + random.choices(self.class_to_indices[c], k=self.k - len(pool))
                    class_to_indices[c] = []

                batch.extend(selected)

                if len(class_to_indices[c]) == 0 and c in available:
                    available.remove(c)

            yield batch

    def __len__(self) -> int:
        return len(sum(self.class_to_indices.values(), [])) // self.batch_size
