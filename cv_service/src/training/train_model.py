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
def prepare_data(batch_size=BATCH_SIZE):
    """Prepare DataLoaders for training and validation"""

    print("* PREPARING DATA *")
    image_paths = []
    labels = []

    for writer in ALL_WRITERS:
        writer_dir = os.path.join(SLIPS_DIR, writer)
        if not os.path.exists(writer_dir):
            continue
        for fname in os.listdir(writer_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(writer_dir, fname)
                test_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if test_img is not None:
                    image_paths.append(path)
                    labels.append(WRITER_TO_ID[writer])

    if len(image_paths) < 20:
        raise ValueError("Not enough data for training")

    # Stratified train/val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"  Training samples: {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")

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

    train_dataset = HandwritingDataset(
        train_paths, train_labels, train_transform)
    val_dataset = HandwritingDataset(val_paths, val_labels, val_transform)

    # Use multiple workers for faster loading
    num_workers = min(8, os.cpu_count() // 2)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


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
def train_efficientnet(train_loader, val_loader, device):
    print("* INITIALISING EFFICIENTNET-B0 MODEL *")
    model = EfficientNetClassifier(num_writers=len(ALL_WRITERS))
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
            model_path = os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, model_path)

            # Sidecar labels file
            sidecar_path = os.path.join(
                MODEL_SAVE_PATH, BEST_MODEL_NAME.replace(".pt", ".labels.json"))
            payload = {
                "all_writers": list(ALL_WRITERS),
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
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, best_val_acc


# ----------------------------
# Main entry
# ----------------------------
def train_supervised():
    print("=== BETFRED HANDWRITING TRAINING (EfficientNet) ===")
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = prepare_data()
    model, val_acc = train_efficientnet(train_loader, val_loader, device)

    print(f"\nTraining completed. Best validation accuracy: {val_acc:.2f}%")
    return model, val_loader, device


if __name__ == "__main__":
    model, val_loader, device = train_supervised()
