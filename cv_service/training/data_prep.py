"""
Data preparation utilities for handwriting classification.

Exposes:
- ApplyCLAHE: optional contrast enhancement transform for grayscale images.
- HandwritingDataset: simple dataset reading grayscale images.
- prepare_data: returns (train_loader, val_loader, writer_list).
"""
from __future__ import annotations

import os
import sys
from typing import List

import cv2
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SLIPS_DIR,
    ALL_WRITERS,
    IMAGE_SIZE,
    BATCH_SIZE,
    PREPROCESS_CLAHE,
)


class ApplyCLAHE:
    """Apply OpenCV CLAHE to a numpy grayscale image if enabled."""

    def __init__(self, enabled: bool = True, clip_limit: float = 2.0, tile_grid_size=(8, 8)):
        self.enabled = enabled
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return img_np
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=self.tile_grid_size)
        return clahe.apply(img_np)


class HandwritingDataset(Dataset):
    """Dataset for handwriting classification."""

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label


def _discover_writers_from_dir(base_dir: str) -> List[str]:
    """Return sorted list of subfolder names as writer classes, or empty if none."""
    if not os.path.isdir(base_dir):
        return []
    names = [d for d in os.listdir(
        base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return sorted(names)


def prepare_data(batch_size: int = BATCH_SIZE):
    """Prepare DataLoaders for training and validation.

    Behavior:
    - If cv_service/dataset_splits/{train,val} exists, load from there (fixed split).
    - Else, read all images under SLIPS_DIR/<writer> and stratify into train/val.
    - Writer classes are discovered from folders if available; otherwise fall back to ALL_WRITERS.
    Returns: (train_loader, val_loader, writer_list)
    """

    # Prefer pre-split dataset if present
    base_dir = os.path.abspath(os.path.join(SLIPS_DIR, os.pardir))
    split_dir = os.path.join(base_dir, "dataset_splits")
    train_split = os.path.join(split_dir, "train")
    val_split = os.path.join(split_dir, "val")

    # Transforms
    train_transform = T.Compose([
        ApplyCLAHE(enabled=PREPROCESS_CLAHE),
        T.ToPILImage(),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomRotation(5),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        ApplyCLAHE(enabled=PREPROCESS_CLAHE),
        T.ToPILImage(),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if os.path.isdir(train_split) and os.path.isdir(val_split):
        writer_list = _discover_writers_from_dir(train_split)
        if not writer_list:
            raise ValueError(f"No writer folders found under {train_split}")

        writer_to_id = {w: i for i, w in enumerate(writer_list)}

        def _gather(split_dir_path):
            paths, lbls = [], []
            for w in writer_list:
                wdir = os.path.join(split_dir_path, w)
                if not os.path.isdir(wdir):
                    continue
                for fname in os.listdir(wdir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        paths.append(os.path.join(wdir, fname))
                        lbls.append(writer_to_id[w])
            return paths, lbls

        train_paths, train_labels = _gather(train_split)
        val_paths, val_labels = _gather(val_split)
    else:
        # Discover writers from SLIPS_DIR, fallback to ALL_WRITERS
        discovered = _discover_writers_from_dir(SLIPS_DIR)
        writer_list = discovered if discovered else list(ALL_WRITERS)
        writer_to_id = {w: i for i, w in enumerate(writer_list)}

        image_paths, labels = [], []
        for writer in writer_list:
            writer_dir = os.path.join(SLIPS_DIR, writer)
            if not os.path.exists(writer_dir):
                continue
            for fname in os.listdir(writer_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    path = os.path.join(writer_dir, fname)
                    test_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if test_img is not None:
                        image_paths.append(path)
                        labels.append(writer_to_id[writer])

        if len(image_paths) < 20:
            raise ValueError("Not enough data for training")

        # Stratified train/val split (30% validation)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.3, random_state=42, stratify=labels
        )

    train_dataset = HandwritingDataset(
        train_paths, train_labels, train_transform)
    val_dataset = HandwritingDataset(val_paths, val_labels, val_transform)

    num_workers = 0
    pin_mem = False
    persistent = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, writer_list
