#!/usr/bin/env python
"""
Main script to train the ensemble handwriting classification model
"""
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

# Import the training function
try:
    from src.training.train_model import train_supervised_model
except ImportError as e:
    print(f"Error importing training module: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("\n🔹 BetFred Handwriting Classification System 🔹")
    print("🔹 Ensemble Model Training Tool           🔹")
    print("🔹 Version 2.0 - July 2025                🔹\n")

    # Train the ensemble model
    train_supervised_model()

    print("\n✅ Training completed successfully")
    print("Run the classification API to use the new ensemble model")
