"""
Training script for the ensemble model - creates placeholder models for demo purposes.
"""
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

try:
    # Import your models
    from src.models.resnet_classifier import ResNetClassifier
    from src.models.efficientnet_classifier import EfficientNetClassifier
    from src.models.densenet_classifier import DenseNetClassifier
    from src.models.ensemble_classifier import EnsembleHandwritingClassifier
    from src.utils.config import MODEL_SAVE_PATH, ALL_WRITERS
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure your model files are in the correct location")
    sys.exit(1)


def create_dummy_models():
    """Create dummy models for demonstration purposes"""
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Number of writers
    num_writers = len(ALL_WRITERS)

    # Create individual models
    models = {
        "resnet": ResNetClassifier(num_writers=num_writers),
        "efficientnet": EfficientNetClassifier(num_writers=num_writers),
        "densenet": DenseNetClassifier(num_writers=num_writers)
    }

    # Save each model
    for name, model in models.items():
        # Initialize with random weights
        model.eval()

        # Path for saving model
        save_path = os.path.join(
            MODEL_SAVE_PATH, f"best_{name}_classifier.pth")

        try:
            # Save the model
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ Created {name} model at {save_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save {name} model: {e}")

    # Try to copy ResNet model to legacy path if it doesn't exist
    legacy_path = os.path.join(
        MODEL_SAVE_PATH, "best_handwriting_classifier.pth")
    resnet_path = os.path.join(MODEL_SAVE_PATH, "best_resnet_classifier.pth")

    if not os.path.exists(legacy_path) and os.path.exists(resnet_path):
        try:
            # Just copy the file
            import shutil
            shutil.copy2(resnet_path, legacy_path)
            logger.info(f"✅ Copied ResNet model to legacy path {legacy_path}")
        except Exception as e:
            logger.error(f"❌ Failed to copy to legacy path: {e}")

    logger.info(f"✅ All dummy models created successfully!")
    logger.info(f"  Models directory: {MODEL_SAVE_PATH}")
    logger.info(f"  Number of writers: {num_writers}")

    # Load the ensemble model to verify it works
    try:
        ensemble = EnsembleHandwritingClassifier(num_writers=num_writers)
        ensemble.load_individual_models()
        logger.info(f"✅ Successfully loaded ensemble model")

        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = ensemble(dummy_input)
        logger.info(f"✅ Ensemble model inference successful")
    except Exception as e:
        logger.error(f"❌ Failed to load ensemble model: {e}")


if __name__ == "__main__":
    logger.info("📊 Creating dummy models for demonstration...")
    create_dummy_models()
