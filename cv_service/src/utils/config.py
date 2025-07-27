"""
Configuration file for supervised handwriting classification
"""
import os

# All 13 writers from your dataset
ALL_WRITERS = [
    'ange', 'fola', 'gina', 'ibra', 'mae', 'mayo', 'nick',
    'odosa', 'sam', 'scott', 'sipo', 'som', 'steve'
]

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
SLIPS_DIR = "slips"
MODEL_SAVE_PATH = "trained_models"
BEST_MODEL_NAME = "best_efficientnet_classifier.pth"

# Create model directory if it doesn't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Performance thresholds
MIN_ACCURACY_THRESHOLD = 0.75
HIGH_CONFIDENCE_THRESHOLD = 0.9
MEDIUM_CONFIDENCE_THRESHOLD = 0.6

print(f"Configuration loaded:")
print(f"  Writers: {len(ALL_WRITERS)}")
print(f"  Expected accuracy: >{MIN_ACCURACY_THRESHOLD:.1%}")
print(f"  Model save path: {MODEL_SAVE_PATH}")
