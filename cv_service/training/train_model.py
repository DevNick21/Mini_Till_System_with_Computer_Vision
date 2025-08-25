"""
Training script for EfficientNet-based handwriting classification
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_SAVE_PATH,
    BEST_MODEL_NAME,
    NUM_EPOCHS,
    LEARNING_RATE,
    EARLY_STOP_PATIENCE,
    SCHEDULER_PATIENCE,
)
from models import EfficientNetClassifier
from training.data_prep import prepare_data as prepare_data_loaders
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _calculate_accuracy(outputs, labels):
    with torch.no_grad():
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).float().sum()
        return correct.item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0.0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += _calculate_accuracy(outputs, labels)
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * total_correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0.0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += _calculate_accuracy(outputs, labels)
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * total_correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def _build_optimizer(model):
    # AdamW with weight decay, excluding biases and norm layers
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    param_groups = [
        {"params": decay, "weight_decay": 1e-4},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return optim.AdamW(param_groups, lr=LEARNING_RATE)


def train_efficientnet(train_loader, val_loader, writer_list, device):
    print("* INITIALISING EFFICIENTNET-B0 MODEL *")
    model = EfficientNetClassifier(
        num_writers=len(writer_list), use_pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=SCHEDULER_PATIENCE, verbose=True)

    best_val_acc = 0.0
    patience_counter = 0
    patience = EARLY_STOP_PATIENCE

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_acc)

        epoch_time = time.time() - start_time
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            model_path = os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME)
            torch.save(model.state_dict(), model_path)

            base_name, _ext = os.path.splitext(BEST_MODEL_NAME)
            ckpt_path = os.path.join(MODEL_SAVE_PATH, f"{base_name}.ckpt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )

            print(f"✅ New best model saved: Val Acc = {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered after {epoch+1} epochs")
                break

    model_path = os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)


    return model, best_val_acc


def train_supervised():
    print("=== HANDWRITING TRAINING ===")
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, writer_list = prepare_data_loaders()
    model, val_acc = train_efficientnet(
        train_loader, val_loader, writer_list, device)

    print(f"\nTraining completed. Best validation accuracy: {val_acc:.2f}%")
    return model, val_loader, device


if __name__ == "__main__":
    model, val_loader, device = train_supervised()
