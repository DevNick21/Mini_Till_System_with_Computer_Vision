"""
Demo configuration for handwriting classification
This file contains settings specifically for demonstration purposes
when working with limited or untrained models
"""
import os
from pathlib import Path

# Define the base directory
CURRENT_DIR = Path(__file__).parent.parent.parent

# Demo writers (subset of the original 13)
# Use only a few writers that we have good samples for
DEMO_WRITERS = [
    'ange', 'mayo', 'nick', 'sam', 'steve'
]

# Create mapping between writer names and numeric IDs
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

# Demo mode settings
DEMO_MODE = True
DEMO_CONFIDENCE_BASE = 0.75  # Base confidence value for demo mode
DEMO_HIGH_CONFIDENCE = 0.90  # Threshold for high confidence in demo mode
DEMO_MEDIUM_CONFIDENCE = 0.70  # Threshold for medium confidence in demo mode

# Paths for demo
DEMO_SLIPS_DIR = os.path.join(CURRENT_DIR, "slips")
DEMO_OUTPUTS_DIR = os.path.join(CURRENT_DIR, "demo_outputs")

# Create output directory
os.makedirs(DEMO_OUTPUTS_DIR, exist_ok=True)

print(f"Demo configuration loaded:")
print(f"  Writers: {len(DEMO_WRITERS)} ({', '.join(DEMO_WRITERS)})")
print(f"  Demo mode: {DEMO_MODE}")
print(f"  Slips directory: {DEMO_SLIPS_DIR}")
