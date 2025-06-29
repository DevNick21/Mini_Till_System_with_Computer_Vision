"""
Detailed evaluation of the trained handwriting classifier
"""
# Set matplotlib backend BEFORE any other imports that might use it
import sys
import pandas as pd
import json
from collections import defaultdict, Counter
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import numpy as np
import cv2
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch
from ..utils.config import *
from ..models.handwriting_classifier import HandwritingClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import our modules
sys.path.append('..')


class EvaluationDataset:
    """Simple dataset for evaluation"""

    def __init__(self, slips_dir):
        self.slips_dir = slips_dir
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Load all images
        self.image_paths = []
        self.labels = []
        self.writer_names = []

        for writer in ALL_WRITERS:
            writer_path = os.path.join(slips_dir, writer)
            if os.path.exists(writer_path):
                for img_file in os.listdir(writer_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(writer_path, img_file)
                        test_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if test_img is not None:
                            self.image_paths.append(img_path)
                            self.labels.append(WRITER_TO_ID[writer])
                            self.writer_names.append(writer)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = self.transform(img)

        return img, self.labels[idx], self.writer_names[idx], img_path


def load_trained_model(model_path):
    """Load the trained model"""
    model = HandwritingClassifier()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def evaluate_model_detailed():
    """Comprehensive model evaluation"""

    print("🔍 DETAILED MODEL EVALUATION")
    print("=" * 50)

    # Load model
    model_path = os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please run training first!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(model_path)
    model.to(device)

    print(f"✅ Model loaded from {model_path}")
    print(f"   Device: {device}")

    # Create evaluation dataset
    eval_dataset = EvaluationDataset(SLIPS_DIR)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    print(f"   Total samples: {len(eval_dataset)}")

    # Collect predictions
    all_predictions = []
    all_labels = []
    all_writer_names = []
    all_confidences = []
    prediction_details = []

    print(f"\n=== RUNNING INFERENCE ===")

    with torch.no_grad():
        for i, (images, labels, writer_names, img_paths) in enumerate(eval_loader):
            images = images.to(device)

            # Get predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            # Store results
            pred_id = predicted.item()
            true_id = labels.item()
            conf_score = confidence.item()

            all_predictions.append(pred_id)
            all_labels.append(true_id)
            all_writer_names.append(writer_names[0])
            all_confidences.append(conf_score)

            # Detailed prediction info
            prediction_details.append({
                'image_path': img_paths[0],
                'true_writer': writer_names[0],
                'predicted_writer': ID_TO_WRITER[pred_id],
                'confidence': conf_score,
                'correct': pred_id == true_id,
                'true_id': true_id,
                'predicted_id': pred_id
            })

            if i % 20 == 0:
                print(f"  Processed {i+1}/{len(eval_dataset)} samples...")

    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_labels, all_predictions)

    print(f"\n=== OVERALL PERFORMANCE ===")
    print(
        f"Overall Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    print(f"Average Confidence: {np.mean(all_confidences):.3f}")
    print(f"Confidence Std: {np.std(all_confidences):.3f}")

    # Per-writer analysis
    print(f"\n=== PER-WRITER PERFORMANCE ===")
    writer_stats = defaultdict(
        lambda: {'correct': 0, 'total': 0, 'confidences': []})

    for i, detail in enumerate(prediction_details):
        writer = detail['true_writer']
        writer_stats[writer]['total'] += 1
        writer_stats[writer]['confidences'].append(detail['confidence'])
        if detail['correct']:
            writer_stats[writer]['correct'] += 1

    writer_performance = []
    for writer in ALL_WRITERS:
        if writer in writer_stats:
            stats = writer_stats[writer]
            accuracy = stats['correct'] / \
                stats['total'] if stats['total'] > 0 else 0
            avg_conf = np.mean(stats['confidences']
                               ) if stats['confidences'] else 0

            writer_performance.append({
                'writer': writer,
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total'],
                'avg_confidence': avg_conf
            })

            print(
                f"  {writer:6s}: {accuracy:.3f} ({stats['correct']:2d}/{stats['total']:2d}) conf={avg_conf:.3f}")
        else:
            print(f"  {writer:6s}: No samples found")

    # Confidence analysis
    print(f"\n=== CONFIDENCE ANALYSIS ===")
    high_conf = sum(1 for c in all_confidences if c >=
                    HIGH_CONFIDENCE_THRESHOLD)
    med_conf = sum(1 for c in all_confidences if MEDIUM_CONFIDENCE_THRESHOLD <=
                   c < HIGH_CONFIDENCE_THRESHOLD)
    low_conf = sum(1 for c in all_confidences if c <
                   MEDIUM_CONFIDENCE_THRESHOLD)

    print(
        f"High confidence (≥{HIGH_CONFIDENCE_THRESHOLD}): {high_conf:3d} ({high_conf/len(all_confidences)*100:.1f}%)")
    print(
        f"Med confidence (≥{MEDIUM_CONFIDENCE_THRESHOLD}): {med_conf:3d} ({med_conf/len(all_confidences)*100:.1f}%)")
    print(
        f"Low confidence (<{MEDIUM_CONFIDENCE_THRESHOLD}): {low_conf:3d} ({low_conf/len(all_confidences)*100:.1f}%)")

    # High confidence accuracy
    high_conf_correct = sum(1 for detail in prediction_details
                            if detail['confidence'] >= HIGH_CONFIDENCE_THRESHOLD and detail['correct'])
    high_conf_accuracy = high_conf_correct / high_conf if high_conf > 0 else 0

    print(
        f"High confidence accuracy: {high_conf_accuracy:.3f} ({high_conf_accuracy*100:.1f}%)")

    # Error analysis
    print(f"\n=== ERROR ANALYSIS ===")
    errors = [detail for detail in prediction_details if not detail['correct']]
    print(f"Total errors: {len(errors)}")

    if errors:
        print(f"Most common confusions:")
        confusion_pairs = defaultdict(int)
        for error in errors:
            pair = f"{error['true_writer']} → {error['predicted_writer']}"
            confusion_pairs[pair] += 1

        for pair, count in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {pair}: {count} times")

    # Save detailed results
    results = {
        'overall_accuracy': overall_accuracy,
        'average_confidence': float(np.mean(all_confidences)),
        'confidence_distribution': {
            'high': high_conf,
            'medium': med_conf,
            'low': low_conf
        },
        'high_confidence_accuracy': high_conf_accuracy,
        'writer_performance': writer_performance,
        'total_samples': len(all_labels),
        'model_path': model_path
    }

    results_path = os.path.join(MODEL_SAVE_PATH, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Detailed results saved to: {results_path}")

    return results, prediction_details


def analyze_similar_writers(prediction_details):
    """Find which writers are most often confused"""

    print(f"\n=== WRITER SIMILARITY ANALYSIS ===")

    # Create confusion matrix data
    confusion_counts = defaultdict(lambda: defaultdict(int))

    for detail in prediction_details:
        true_writer = detail['true_writer']
        pred_writer = detail['predicted_writer']
        confusion_counts[true_writer][pred_writer] += 1

    # Find most confused pairs
    confusion_pairs = []
    for true_writer in ALL_WRITERS:
        for pred_writer in ALL_WRITERS:
            if true_writer != pred_writer:
                count = confusion_counts[true_writer][pred_writer]
                if count > 0:
                    confusion_pairs.append((true_writer, pred_writer, count))

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    print("Most confused writer pairs:")
    for true_w, pred_w, count in confusion_pairs[:10]:
        print(f"  {true_w} confused as {pred_w}: {count} times")

    return confusion_pairs


def generate_betfred_visualizations(all_labels, all_predictions, all_confidences, writer_performance, save_dir="trained_models", all_writers=ALL_WRITERS):
    """
    Generate all key evaluation plots with Betfred branding.
    """

    # Define Betfred color palette
    BETFRED_BLUE = "#0033A0"
    BETFRED_RED = "#EE2737"
    BETFRED_WHITE = "#FFFFFF"
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['figure.facecolor'] = BETFRED_WHITE

    os.makedirs(save_dir, exist_ok=True)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=sns.color_palette([BETFRED_BLUE, BETFRED_RED], as_cmap=True),
        xticklabels=all_writers,
        yticklabels=all_writers,
        cbar=False,
        annot_kws={"size": 12, "weight": "bold", "color": BETFRED_WHITE}
    )
    plt.title("Confusion Matrix: Handwriting Classifier",
              color=BETFRED_BLUE, fontsize=16, weight='bold')
    plt.xlabel("Predicted Writer", color=BETFRED_RED, weight='bold')
    plt.ylabel("True Writer", color=BETFRED_RED, weight='bold')
    plt.xticks(rotation=45, ha="right", color=BETFRED_BLUE)
    plt.yticks(rotation=0, color=BETFRED_BLUE)
    plt.tight_layout()
    plt.savefig(os.path.join(
        save_dir, "confusion_matrix_betfred.png"), dpi=300)
    plt.close()

    # ---- Per-Writer Accuracy Bar Plot ----
    writers = [wp['writer'] for wp in writer_performance]
    accuracies = [wp['accuracy'] for wp in writer_performance]
    plt.figure(figsize=(11, 5))
    bars = sns.barplot(x=writers, y=accuracies, palette=[
                       BETFRED_RED if acc < 0.7 else BETFRED_BLUE for acc in accuracies])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy", color=BETFRED_RED, weight='bold')
    plt.xlabel("Writer", color=BETFRED_RED, weight='bold')
    plt.title("Per-Writer Classification Accuracy",
              color=BETFRED_BLUE, fontsize=16, weight='bold')
    plt.xticks(color=BETFRED_BLUE)
    plt.yticks(color=BETFRED_BLUE)
    for bar, acc in zip(bars.patches, accuracies):
        plt.annotate(f"{acc:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom', color=BETFRED_RED if acc < 0.7 else BETFRED_BLUE, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(
        save_dir, "per_writer_accuracy_betfred.png"), dpi=300)
    plt.close()

    # ---- Confidence Histogram ----
    plt.figure(figsize=(8, 5))
    plt.hist(all_confidences, bins=20, color=BETFRED_BLUE,
             alpha=0.85, edgecolor=BETFRED_RED)
    plt.xlabel("Confidence Score", color=BETFRED_BLUE, weight='bold')
    plt.ylabel("Count", color=BETFRED_RED, weight='bold')
    plt.title("Prediction Confidence Distribution",
              color=BETFRED_BLUE, fontsize=15, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(
        save_dir, "confidence_histogram_betfred.png"), dpi=300)
    plt.close()

    # ---- Confidence Category Bar Chart ----
    conf_cat = ['High' if c >= HIGH_CONFIDENCE_THRESHOLD else 'Medium' if c >= MEDIUM_CONFIDENCE_THRESHOLD else 'Low'
                for c in all_confidences]
    cat_counts = Counter(conf_cat)
    plt.figure(figsize=(5, 5))
    bar_palette = [BETFRED_BLUE if cat == "High" else BETFRED_RED if cat ==
                   "Low" else "#888888" for cat in cat_counts.keys()]
    sns.barplot(x=list(cat_counts.keys()), y=list(
        cat_counts.values()), palette=bar_palette)
    plt.ylabel("Number of Predictions", color=BETFRED_RED, weight='bold')
    plt.xlabel("Confidence Category", color=BETFRED_RED, weight='bold')
    plt.title("Confidence Category Distribution",
              color=BETFRED_BLUE, fontsize=15, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(
        save_dir, "confidence_categories_betfred.png"), dpi=300)
    plt.close()

    # ---- Save confusion matrix as CSV ----
    cm_df = pd.DataFrame(cm, index=all_writers, columns=all_writers)
    cm_df.to_csv(os.path.join(save_dir, "confusion_matrix_betfred.csv"))

    print("\n✅ Betfred-branded evaluation visuals saved in:",
          os.path.abspath(save_dir))


if __name__ == "__main__":
    results, details = evaluate_model_detailed()
    confusion_pairs = analyze_similar_writers(details)
    all_labels = [d['true_id'] for d in details]
    all_predictions = [d['predicted_id'] for d in details]
    all_confidences = [d['confidence'] for d in details]
    writer_performance = results['writer_performance']

    # ---- GENERATE ALL VISUALS ----
    generate_betfred_visualizations(
        all_labels,
        all_predictions,
        all_confidences,
        writer_performance,
        save_dir=MODEL_SAVE_PATH,
        all_writers=ALL_WRITERS
    )

    print(f"\n🎯 SUMMARY:")
    print(f"   Current accuracy: {results['overall_accuracy']:.1%}")
    print(
        f"   High confidence cases: {results['confidence_distribution']['high']} ({results['confidence_distribution']['high']/results['total_samples']*100:.1f}%)")
    print(
        f"   High confidence accuracy: {results['high_confidence_accuracy']:.1%}")
