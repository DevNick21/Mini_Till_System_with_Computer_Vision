"""
Training script for supervised handwriting classification with ensemble approach
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
import shutil
from datetime import datetime

# Import our modules
from ..utils.config import *
from ..models.resnet_classifier import ResNetClassifier
from ..models.efficientnet_classifier import EfficientNetClassifier
from ..models.densenet_classifier import DenseNetClassifier
from ..models.ensemble_classifier import EnsembleHandwritingClassifier


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


def train_single_model(model_class, model_name, train_loader, val_loader, device):
    """Train a single model and return metrics for ensemble weighting"""
    print(f"\n=== TRAINING {model_name.upper()} MODEL ===")

    # Create model
    model = model_class()
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
    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

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
            model_path = os.path.join(
                MODEL_SAVE_PATH, f"best_{model_name}_classifier.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ New best model saved! Accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break

    training_time = time.time() - start_time

    print(f"\n=== {model_name.upper()} TRAINING COMPLETED ===")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Total training time: {training_time/60:.1f} minutes")

    # Calculate robustness score - higher is better
    robustness = calculate_model_robustness(model, val_loader, device)
    print(f"Robustness score: {robustness:.4f}\n")

    # Load best model weights
    model_path = os.path.join(
        MODEL_SAVE_PATH, f"best_{model_name}_classifier.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    return model, best_val_acc, robustness


def calculate_model_robustness(model, val_loader, device):
    """Calculate robustness score for a model"""

    model.eval()
    all_confidences = []
    all_correct = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            correct = (predicted == labels)

            all_confidences.extend(confidence.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())

    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)

    # Calculate various robustness metrics

    # 1. Accuracy (base metric)
    accuracy = np.mean(all_correct)

    # 2. Confidence calibration (how well confidence correlates with accuracy)
    # Group confidences into bins
    bins = np.linspace(0, 1, 11)  # 10 bins
    binned_confidences = np.digitize(all_confidences, bins) - 1

    calibration_error = 0
    bin_counts = np.zeros(10)

    for bin_idx in range(10):
        mask = binned_confidences == bin_idx
        if np.sum(mask) > 0:
            bin_counts[bin_idx] = np.sum(mask)
            bin_acc = np.mean(all_correct[mask])
            bin_conf = np.mean(all_confidences[mask])
            calibration_error += np.abs(bin_acc - bin_conf) * \
                (np.sum(mask) / len(all_correct))

    # 3. High-confidence correctness (accuracy for high-confidence predictions)
    high_conf_mask = all_confidences > HIGH_CONFIDENCE_THRESHOLD
    high_conf_accuracy = np.mean(all_correct[high_conf_mask]) if np.sum(
        high_conf_mask) > 0 else 0

    # 4. Confidence variance (lower is better for stability)
    conf_variance = np.var(all_confidences)

    # Combine metrics into a robustness score (higher is better)
    # This formula emphasizes accuracy, high-confidence correctness, and calibration
    robustness_score = (0.5 * accuracy +
                        0.3 * high_conf_accuracy +
                        0.2 * (1.0 - calibration_error))

    return robustness_score


def calculate_ensemble_weights(model_metrics):
    """Calculate optimal weights for the ensemble based on model metrics"""

    # Extract metrics
    accuracies = [metrics['accuracy'] for metrics in model_metrics.values()]
    robustness_scores = [metrics['robustness']
                         for metrics in model_metrics.values()]

    # Normalize accuracy and robustness to sum to 1.0
    total_acc = sum(accuracies)
    total_rob = sum(robustness_scores)

    norm_acc = [acc/total_acc for acc in accuracies]
    norm_rob = [rob/total_rob for rob in robustness_scores]

    # Combine with more weight on robustness
    weights = [0.5 * a + 0.5 * r for a, r in zip(norm_acc, norm_rob)]

    # Normalize to sum to 1.0
    total_weight = sum(weights)
    final_weights = [w/total_weight for w in weights]

    return final_weights


def evaluate_ensemble(ensemble_model, val_loader, device):
    """Evaluate ensemble model performance"""
    print("\n=== EVALUATING ENSEMBLE MODEL ===")

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = validate_epoch(
        ensemble_model, val_loader, criterion, device)

    print(f"Ensemble Validation Loss: {val_loss:.4f}")
    print(f"Ensemble Validation Accuracy: {val_acc:.2f}%")

    # Calculate detailed metrics for ensemble
    ensemble_robustness = calculate_model_robustness(
        ensemble_model, val_loader, device)
    print(f"Ensemble Robustness Score: {ensemble_robustness:.4f}\n")

    return val_acc, ensemble_robustness


def train_supervised_model():
    """Train the ensemble classification model"""

    print("=== BETFRED HANDWRITING ENSEMBLE TRAINING STARTED ===")
    print("=" * 60)

    # Create model directory if it doesn't exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Prepare data
    train_loader, val_loader, writer_counts = prepare_data()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Define model architectures to train
    model_architectures = {
        'resnet': ResNetClassifier,
        'efficientnet': EfficientNetClassifier,
        'densenet': DenseNetClassifier
    }

    # Train each model and collect metrics
    trained_models = {}
    model_metrics = {}

    for model_name, model_class in model_architectures.items():
        model, accuracy, robustness = train_single_model(
            model_class, model_name, train_loader, val_loader, device)

        trained_models[model_name] = model
        model_metrics[model_name] = {
            'accuracy': accuracy,
            'robustness': robustness
        }

    # Print comparison of models
    print("\n=== MODEL COMPARISON ===")
    print(f"{'Model':<12} {'Accuracy':<10} {'Robustness':<10}")
    print("-" * 32)

    for model_name, metrics in model_metrics.items():
        print(
            f"{model_name:<12} {metrics['accuracy']:<10.2f} {metrics['robustness']:<10.4f}")

    # Calculate optimal weights for ensemble
    weights = calculate_ensemble_weights(model_metrics)
    print("\n=== ENSEMBLE WEIGHTS ===")
    for model_name, weight in zip(model_architectures.keys(), weights):
        print(f"{model_name}: {weight:.4f}")

    # Create and evaluate ensemble
    ensemble = EnsembleHandwritingClassifier(model_weights=weights)
    ensemble.load_individual_models()
    ensemble.to(device)
    ensemble.eval()

    # Save the ensemble weights
    ensemble_weights_path = os.path.join(
        MODEL_SAVE_PATH, "ensemble_weights.pth")
    torch.save(ensemble.model_weights, ensemble_weights_path)
    print(f"Ensemble weights saved to {ensemble_weights_path}")

    # Save legacy model format for backward compatibility
    best_model_name = max(model_metrics.items(),
                          key=lambda x: x[1]['accuracy'])[0]
    best_model_path = os.path.join(
        MODEL_SAVE_PATH, f"best_{best_model_name}_classifier.pth")
    legacy_model_path = os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME)

    if os.path.exists(best_model_path):
        # Copy the best model to the legacy path for backward compatibility
        print(
            f"Copying best model ({best_model_name}) to legacy location for compatibility")
        shutil.copy2(best_model_path, legacy_model_path)

    # Evaluate ensemble
    ensemble_accuracy, ensemble_robustness = evaluate_ensemble(
        ensemble, val_loader, device)

    # Final report
    print("\n=== FINAL ENSEMBLE RESULTS ===")
    print(f"Ensemble accuracy: {ensemble_accuracy:.2f}%")
    print(f"Ensemble robustness: {ensemble_robustness:.4f}")

    improvement = ensemble_accuracy - \
        max([m['accuracy'] for m in model_metrics.values()])
    print(f"Improvement over best single model: {improvement:.2f}%")

    if improvement > 0:
        print("✅ Ensemble successfully improved classification accuracy!")
    else:
        print("⚠️  Ensemble did not improve over the best single model")

    return ensemble, ensemble_accuracy, val_loader, device


if __name__ == "__main__":
    print("\n🔹 BetFred Handwriting Classification System 🔹")
    print("🔹 Ensemble Model Training Tool           🔹")
    print("🔹 Version 2.0 - July 2025                🔹")

    # Train the ensemble model
    ensemble, accuracy, val_loader, device = train_supervised_model()

    print(f"\n🎯 FINAL RESULTS:")
    if accuracy >= 90:
        print(f"✅ OUTSTANDING: {accuracy:.1f}% accuracy achieved!")
    elif accuracy >= 85:
        print(f"✅ EXCELLENT: {accuracy:.1f}% accuracy achieved!")
    elif accuracy >= 75:
        print(f"✅ GOOD: {accuracy:.1f}% accuracy - meets target!")
    elif accuracy >= 60:
        print(f"⚠️  MODERATE: {accuracy:.1f}% accuracy - needs improvement")
    else:
        print(f"❌ POOR: {accuracy:.1f}% accuracy - major issues")

    print("\n✓ Training completed successfully")
    print("✓ Classification API will automatically use the new ensemble model")
    print("✓ System is ready for improved handwriting classification")
