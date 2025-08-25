import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .efficientnet_classifier import EfficientNetClassifier  # noqa: F401
