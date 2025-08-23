"""
Configuration for supervised handwriting classification (production)
"""
import os
from pathlib import Path

# Base directory for cv_service
CURRENT_DIR = Path(__file__).parent

# All 12 writers
ALL_WRITERS = ["anon" + str(i + 1) for i in range(12)]

# Create mapping between writer names and numeric IDs
WRITER_TO_ID = {writer: idx for idx, writer in enumerate(ALL_WRITERS)}
ID_TO_WRITER = {idx: writer for idx, writer in enumerate(ALL_WRITERS)}


# Training parameters
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
IMAGE_SIZE = 224
DROPOUT_RATE = 0.3

# Data paths
SLIPS_DIR = os.path.join(CURRENT_DIR, "slips")
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "trained_models")
BEST_MODEL_NAME = "best_efficientnet_classifier.pth"

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Performance thresholds
MIN_ACCURACY_THRESHOLD = 0.75
HIGH_CONFIDENCE_THRESHOLD = 0.9
MEDIUM_CONFIDENCE_THRESHOLD = 0.75

print(f"Configuration loaded:")
print(f"  All Writers: {len(ALL_WRITERS)}")
print(f"  Expected accuracy: >{MIN_ACCURACY_THRESHOLD:.1%}")
print(f"  Model save path: {MODEL_SAVE_PATH}")
print(f"  Slips directory: {SLIPS_DIR}")

# Inference preprocessing toggles
# Keep this aligned with training transforms to avoid distribution shift
PREPROCESS_CLAHE = True  # If True, apply CLAHE consistently
