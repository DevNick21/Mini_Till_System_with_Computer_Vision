"""
BetFred CV Service Utils Module
Provides utility functions and configuration for the CV service
"""
from .config import *

__all__ = [
    'ALL_WRITERS', 'WRITER_TO_ID', 'ID_TO_WRITER',
    'BATCH_SIZE', 'LEARNING_RATE', 'NUM_EPOCHS', 'IMAGE_SIZE', 'DROPOUT_RATE',
    'SLIPS_DIR', 'MODEL_SAVE_PATH', 'BEST_MODEL_NAME',
    'HIGH_CONFIDENCE_THRESHOLD', 'MEDIUM_CONFIDENCE_THRESHOLD',
    'MIN_ACCURACY_THRESHOLD', 'PREPROCESS_CLAHE'
]
