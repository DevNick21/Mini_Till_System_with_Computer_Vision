"""Core modules for classification service.

Exposes inference utilities for API layer.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .inference import init_model, classify_image, model_info  # noqa: F401
