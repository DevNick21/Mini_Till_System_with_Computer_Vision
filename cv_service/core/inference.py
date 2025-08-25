
import os
import sys
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from typing import Tuple, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_SAVE_PATH,
    ALL_WRITERS,
    IMAGE_SIZE,
    PREPROCESS_CLAHE,
    MEDIUM_CONFIDENCE_THRESHOLD,
    ID_TO_WRITER,
    BEST_MODEL_NAME,
)
from models import EfficientNetClassifier


# Runtime state
_model = None
_device = None
_transform = None


def init_model():
    global _model, _device, _transform
    if _model is not None:
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    efficientnet_path = os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME)
    if not os.path.exists(efficientnet_path):
        raise FileNotFoundError(
            f"EfficientNet model not found at {efficientnet_path}")

    model = EfficientNetClassifier(num_writers=len(ALL_WRITERS))
    model.load_state_dict(torch.load(efficientnet_path, map_location=device))
    model.to(device)
    model.eval()
    
    _model = model
    _device = device
    _transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def classify_image(image_bytes: bytes) -> Tuple[int, float, str]:
    """Returns (writer_id, confidence, confidence_level)."""
    if _model is None:
        init_model()

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")

    if PREPROCESS_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    img_tensor = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        outputs = _model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_id = torch.max(probabilities, 1)
        writer_id = predicted_id.item() + 1
        confidence_score = confidence.item()

        if confidence_score >= 0.9:
            conf_level = "high"
        elif confidence_score >= MEDIUM_CONFIDENCE_THRESHOLD:
            conf_level = "medium"
        else:
            conf_level = "low"
    return writer_id, confidence_score, conf_level


def model_info() -> Dict:
    if _model is None:
        init_model()
    num_writers = len(ALL_WRITERS)
    writer_ids = list(range(1, num_writers + 1))
    model_type = "EfficientNetClassifier" if isinstance(
        _model, EfficientNetClassifier) else "Unknown"
    return {
        "model_type": model_type,
        "num_writers": num_writers,
        "available_writers": writer_ids,
        "input_size": IMAGE_SIZE,
        "device": str(_device),
        "confidence_thresholds": {"high": 0.9, "medium": MEDIUM_CONFIDENCE_THRESHOLD, "low": 0.0},
        "business_threshold": MEDIUM_CONFIDENCE_THRESHOLD,
    }
