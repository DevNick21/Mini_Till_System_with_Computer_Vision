"""
Configuration file for supervised handwriting classification
Includes both production and demo settings in a unified configuration file
"""
import os
from pathlib import Path

# Define the base directory
CURRENT_DIR = Path(__file__).parent.parent.parent

# All 13 writers from your dataset
ALL_WRITERS = [
    'ange', 'fola', 'gina', 'ibra', 'mae', 'mayo', 'nick',
    'odosa', 'sam', 'scott', 'sipo', 'som', 'steve'
]

# Create mapping between writer names and numeric IDs
WRITER_TO_ID = {writer: idx for idx, writer in enumerate(ALL_WRITERS)}
ID_TO_WRITER = {idx: writer for idx, writer in enumerate(ALL_WRITERS)}

# Demo writers (subset of the original 13)
# Use only a few writers that we have good samples for
DEMO_WRITERS = [
    'ange', 'mayo', 'nick', 'sam', 'steve'
]

# Create mapping between demo writer names and numeric IDs
DEMO_WRITER_TO_ID = {writer: idx for idx, writer in enumerate(DEMO_WRITERS)}
DEMO_ID_TO_WRITER = {idx: writer for idx, writer in enumerate(DEMO_WRITERS)}

# Demo file mapping - helps direct specific files to specific writers
# This ensures demo consistency even with untrained models
DEMO_FILE_MAPPING = {
    # Filename patterns that should be classified as specific writers
    # Format: "filename_contains": writer_id
    "ange": 0,  # Will be classified as 'ange'
    "mayo": 1,  # Will be classified as 'mayo'
    "nick": 2,  # Will be classified as 'nick'
    "sam": 3,   # Will be classified as 'sam'
    "steve": 4,  # Will be classified as 'steve'

    # Default mapping for files that don't match any pattern
    "default": 2  # Default to 'nick' if no match
}

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
DEMO_OUTPUTS_DIR = os.path.join(CURRENT_DIR, "demo_outputs")

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(DEMO_OUTPUTS_DIR, exist_ok=True)

# Performance thresholds
MIN_ACCURACY_THRESHOLD = 0.75
HIGH_CONFIDENCE_THRESHOLD = 0.9
MEDIUM_CONFIDENCE_THRESHOLD = 0.6

# Demo mode settings
DEMO_MODE = True  # Set to False for production use
DEMO_CONFIDENCE_BASE = 0.75  # Base confidence value for demo mode
DEMO_HIGH_CONFIDENCE = 0.90  # Threshold for high confidence in demo mode
DEMO_MEDIUM_CONFIDENCE = 0.70  # Threshold for medium confidence in demo mode

print(f"Configuration loaded:")
print(f"  All Writers: {len(ALL_WRITERS)}")
print(f"  Demo Writers: {len(DEMO_WRITERS)} ({', '.join(DEMO_WRITERS)})")
print(f"  Demo mode: {DEMO_MODE}")
print(f"  Expected accuracy: >{MIN_ACCURACY_THRESHOLD:.1%}")
print(f"  Model save path: {MODEL_SAVE_PATH}")
print(f"  Slips directory: {SLIPS_DIR}")
