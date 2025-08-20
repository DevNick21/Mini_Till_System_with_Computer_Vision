"""
Training script for EfficientNet-based handwriting classification
"""
import os
import random
import time
from datetime import datetime
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from ..models.efficientnet_classifier import EfficientNetClassifier
from ..utils.config import *

# ----------------------------
# Utilities
# ----------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Dataset
# ----------------------------
class HandwritingDataset(Dataset):
    """Dataset for handwriting classification"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224), dtype=np.uint8)
            print(f"Warning: Could not load {img_path}, using black image")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label


# ----------------------------
# Data preparation
# ----------------------------
def _discover_writers_from_dir(base_dir: str):
    """Return sorted list of subfolder names as writer classes, or empty if none."""
    if not os.path.isdir(base_dir):
        return []
    names = [d for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d))]
    return sorted(names)


def prepare_data(batch_size=BATCH_SIZE):
    """Prepare DataLoaders for training and validation.

    Behavior:
    - If cv_service/dataset_splits/{train,val} exists, load from there (fixed split).
    - Else, read all images under SLIPS_DIR/<writer> and stratify into train/val.
    - Writer classes are discovered from folders if available; otherwise fall back to ALL_WRITERS.
    Returns: (train_loader, val_loader, writer_list)
    """

    print("* PREPARING DATA *")

    # Prefer pre-split dataset if present
    base_dir = os.path.abspath(os.path.join(SLIPS_DIR, os.pardir))
    split_dir = os.path.join(base_dir, "dataset_splits")
    train_split = os.path.join(split_dir, "train")
    val_split = os.path.join(split_dir, "val")

    # Transforms
    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomRotation((-5, 5)),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if os.path.isdir(train_split) and os.path.isdir(val_split):
        # Fixed split loading
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
        print(f"  Using pre-split dataset at {split_dir}")
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

        # Stratified train/val split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.3, random_state=42, stratify=labels
        )
        print(f"  Using SLIPS_DIR={SLIPS_DIR}")

    print(f"  Training samples: {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")
    print(
        f"  Writers (classes): {len(writer_list)} -> {', '.join(writer_list)}")

    train_dataset = HandwritingDataset(
        train_paths, train_labels, train_transform)
    val_dataset = HandwritingDataset(val_paths, val_labels, val_transform)

    # Use multiple workers for faster loading
    num_workers = min(8, max(1, (os.cpu_count() or 2) // 2))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, writer_list


# ----------------------------
# Training / Validation
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ----------------------------
# Training loop
# ----------------------------
def train_efficientnet(train_loader, val_loader, writer_list, device):
    print("* INITIALISING EFFICIENTNET-B0 MODEL *")
    model = EfficientNetClassifier(num_writers=len(writer_list))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=5, verbose=True)

    best_val_acc = 0.0
    patience_counter = 0
    patience = 10

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device)

        scheduler.step(val_acc)  # adjust LR based on val accuracy

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save state_dict as primary model file for inference
            model_path = os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME)
            torch.save(model.state_dict(), model_path)

            # Also save a training checkpoint alongside (optional)
            base_name, _ext = os.path.splitext(BEST_MODEL_NAME)
            ckpt_path = os.path.join(MODEL_SAVE_PATH, f"{base_name}.ckpt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)

            # Sidecar labels file with correct name
            sidecar_path = os.path.join(
                MODEL_SAVE_PATH, f"{base_name}.labels.json")
            payload = {
                "all_writers": list(writer_list),
                "created": datetime.utcnow().isoformat() + "Z",
                "image_size": IMAGE_SIZE,
                "arch": "EfficientNet-B0",
            }
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            print(f"✅ New best model saved: Val Acc = {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    model_path = os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME)
    state_dict = torch.load(model_path, map_location=device)
    # state_dict is the model weights saved above
    model.load_state_dict(state_dict)

    return model, best_val_acc


# ----------------------------
# Main entry
# ----------------------------
def train_supervised():
    print("=== BETFRED HANDWRITING TRAINING (EfficientNet) ===")
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, writer_list = prepare_data()
    model, val_acc = train_efficientnet(
        train_loader, val_loader, writer_list, device)

    print(f"\nTraining completed. Best validation accuracy: {val_acc:.2f}%")
    return model, val_loader, device


if __name__ == "__main__":
    model, val_loader, device = train_supervised()
