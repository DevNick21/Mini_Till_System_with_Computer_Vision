"""
Training script for supervised handwriting classification
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import time
from datetime import datetime

# Import our modules
from ..utils.config import *
from ..models.handwriting_classifier import HandwritingClassifier


class HandwritingDataset(Dataset):
    """Dataset for handwriting classification"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Return a black image if the loading fails
            img = np.zeros((224, 224), dtype=np.uint8)
            print(f"Warning: Could not load {img_path}, using black image")

        # Basic preprocessing
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # Apply transforms if specified
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label


def prepare_data():
    """Prepare training and validation datasets"""

    print("* PREPARING SUPERVISED LEARNING DATA *")

    image_paths = []
    labels = []
    writer_counts = {}

    # Collect all image paths and labels
    for writer in ALL_WRITERS:
        writer_path = os.path.join(SLIPS_DIR, writer)
        writer_count = 0

        if os.path.exists(writer_path):
            for img_file in os.listdir(writer_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(writer_path, img_file)

                    # Verify image can be loaded
                    test_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if test_img is not None:
                        image_paths.append(img_path)
                        labels.append(WRITER_TO_ID[writer])
                        writer_count += 1

        writer_counts[writer] = writer_count
        print(f"  {writer}: {writer_count} samples")

    print(f"\nTotal samples: {len(image_paths)}")
    print(f"Writers: {len(set(labels))}")

    if len(image_paths) < 20:
        raise ValueError("Not enough data for training")

    # Data transforms
    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomRotation((-5, 5)),  # Small rotation
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small translation
        T.Grayscale(num_output_channels=3),  # Convert to 3-channel for ResNet
        T.ToTensor(),
        # ImageNet normalization
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Split data (stratified to ensure each writer in both sets)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"\nData split:")
    print(f"  Training: {len(train_paths)} samples")
    print(f"  Validation: {len(val_paths)} samples")

    # Create datasets
    train_dataset = HandwritingDataset(
        train_paths, train_labels, train_transform)
    val_dataset = HandwritingDataset(val_paths, val_labels, val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, writer_counts


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 10 == 0:
            print(
                f"    Batch {batch_idx:3d}: Loss={loss.item():.4f}, Acc={100.*correct/total:.1f}%")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def train_supervised_model():
    """Main training function"""

    print("STARTING TRAINING")
    print("=" * 50)

    # Prepare data
    train_loader, val_loader, writer_counts = prepare_data()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create model
    model = HandwritingClassifier()
    model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    # Training variables
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    print(f"\n=== TRAINING CONFIGURATION ===")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Device: {device}")
    print(f"Writers: {len(ALL_WRITERS)}")

    print(f"\n=== STARTING TRAINING ===")
    start_time = time.time()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        print(f"\nEpoch [{epoch+1:2d}/{NUM_EPOCHS}]")

        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)

        # Validation phase
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print epoch results
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(
            f"  Time: {epoch_time:.1f}s, LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save model
            model_path = os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME)
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ New best model saved! Accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break

    training_time = time.time() - start_time

    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Total training time: {training_time/60:.1f} minutes")
    print(f"Model saved to: {os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME)}")

    # Load best model for final evaluation
    model.load_state_dict(torch.load(
        os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME)))

    return model, best_val_acc, val_loader, device


if __name__ == "__main__":
    model, accuracy, val_loader, device = train_supervised_model()

    print(f"\n🎯 FINAL RESULTS:")
    if accuracy >= 85:
        print(f"✅ EXCELLENT: {accuracy:.1f}% accuracy achieved!")
    elif accuracy >= 75:
        print(f"✅ GOOD: {accuracy:.1f}% accuracy - meets target!")
    elif accuracy >= 60:
        print(f"⚠️  MODERATE: {accuracy:.1f}% accuracy - needs improvement")
    else:
        print(f"❌ POOR: {accuracy:.1f}% accuracy - major issues")
