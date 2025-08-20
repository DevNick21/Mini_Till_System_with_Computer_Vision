"""
Classification API for BetFred Writer Identification System
Provides REST endpoints for handwriting classification with confidence scoring
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Dict, Any
import torch
import torchvision.transforms as T
import cv2
import numpy as np
import logging
from datetime import datetime
from pydantic import BaseModel, ConfigDict
import os
import sys
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

# Import models and config
try:
    # Import models and utils with improved structure
    from src.models import EfficientNetClassifier
    from src.utils import (
        MODEL_SAVE_PATH,
        ALL_WRITERS,
        IMAGE_SIZE,
        PREPROCESS_CLAHE,
        MEDIUM_CONFIDENCE_THRESHOLD,
        ID_TO_WRITER,
        BEST_MODEL_NAME,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure your model and config files are in the correct location")
    sys.exit(1)

# Configure logging without colors
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',  # Simple format without colors
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BetFred Writer Classification API (EfficientNet)",
    version="1.0.0",
    description="Enhanced handwriting classification service using EfficientNet architecture"
)


def to_camel(s: str) -> str:
    """Convert snake_case to camelCase for JSON field names"""
    parts = s.split('_')
    return parts[0] + ''.join(p.title() for p in parts[1:])


class ClassificationResult(BaseModel):
    # Use camelCase aliases for JSON
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    slip_id: int
    writer_id: int          # Numeric writer ID (1-13)
    confidence: float
    confidence_level: str   # "high", "medium", "low"


class ClassificationResponse(BaseModel):
    # Use camelCase aliases for JSON
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    results: List[ClassificationResult]
    summary: Dict[str, Any]
    timestamp: str


# Global model variables
_model = None
_device = None
_transform = None
_class_order = None  # Optional class order loaded from sidecar
_id_to_writer_runtime = None  # Mapping resolved at runtime


def initialize_model():
    """Initialize the trained handwriting classification model (EfficientNet)"""
    global _model, _device, _transform

    try:
        # Determine device
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {_device}")

    # Check if the EfficientNet model is available
        efficientnet_path = os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME)

        # Load EfficientNet model
        if os.path.exists(efficientnet_path):
            try:
                logger.info("EfficientNet model found, using as primary model")
                # Try to load class order sidecar to size model appropriately
                base_name, _ = os.path.splitext(BEST_MODEL_NAME)
                sidecar_path = os.path.join(
                    MODEL_SAVE_PATH, f"{base_name}.labels.json")
                runtime_class_order = None
                if os.path.exists(sidecar_path):
                    try:
                        import json
                        with open(sidecar_path, "r", encoding="utf-8") as f:
                            payload = json.load(f)
                        if isinstance(payload, dict) and "all_writers" in payload and isinstance(payload["all_writers"], list):
                            runtime_class_order = payload["all_writers"]
                            logger.info(
                                f"Loaded class order from sidecar: {sidecar_path}")
                    except Exception as se:
                        logger.warning(f"Failed to read labels sidecar: {se}")

                # Size the classifier using sidecar if present
                num_classes = len(
                    runtime_class_order) if runtime_class_order else len(ALL_WRITERS)
                _model = EfficientNetClassifier(num_writers=num_classes)
                _model.load_state_dict(torch.load(
                    efficientnet_path, map_location=_device))
                _model.to(_device)
                _model.eval()
                model_path = efficientnet_path
                logger.info("EfficientNet model loaded successfully")

                # Resolve class order/name mapping for runtime
                global _class_order, _id_to_writer_runtime
                if runtime_class_order:
                    _class_order = runtime_class_order
                    _id_to_writer_runtime = {
                        idx: name for idx, name in enumerate(runtime_class_order)}
                    # Sanity check vs config
                    if len(runtime_class_order) != len(ALL_WRITERS) or any(a != b for a, b in zip(runtime_class_order, ALL_WRITERS)):
                        logger.warning(
                            "Class order sidecar differs from config ALL_WRITERS; using sidecar order for API name mapping.")
                else:
                    _class_order = list(ALL_WRITERS)
                    _id_to_writer_runtime = dict(ID_TO_WRITER)
            except Exception as e:
                logger.error(f"✗ Error loading EfficientNet model: {e}")
                _model = None
        else:
            raise FileNotFoundError(
                f"EfficientNet model not found in {MODEL_SAVE_PATH}")

            # This section has been moved up above

        # Create image transformation pipeline
        _transform = T.Compose([
            T.ToPILImage(),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        logger.info("Classification model initialized successfully")
        logger.info(f"   Model path: {model_path}")
        logger.info(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
        num_writers = len(_class_order) if _class_order else len(ALL_WRITERS)
        logger.info(
            f"   Writers: {num_writers} (IDs 1-{num_writers})")

    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting BetFred Classification API...")
    initialize_model()
    logger.info("API ready for classification requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # Cleanup code would go here if needed
    pass


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api": "BetFred Writer Classification",
        "model_loaded": _model is not None,
        "device": str(_device) if _device else "unknown",
        "writers": len(_class_order) if _class_order else len(ALL_WRITERS),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/classify-anonymous", response_model=ClassificationResult)
async def classify_handwriting(file: UploadFile = File(...)):
    """Classify a single handwriting slip and return one result"""

    if _model is None:
        raise HTTPException(
            status_code=503, detail="Classification model not loaded")

    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    logger.info(f"Processing slip image for classification: {file.filename}")

    # Extract slip ID from filename (format: "123.jpg")
    slip_id = 1
    if file.filename and '.' in file.filename:
        slip_id_str = file.filename.split('.')[0]
        try:
            slip_id = int(slip_id_str)
            logger.info(f"Extracted SlipId: {slip_id}")
        except ValueError:
            logger.warning(
                f"Could not parse slip ID from filename: {file.filename}; defaulting to 1")

    # Read and decode image
    image_data = await file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(
            status_code=400, detail=f"Could not decode image {file.filename}")

    logger.info(f"Image decoded successfully: {img.shape}")

    # Optional CLAHE, controlled by config; default off to match training
    if PREPROCESS_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    # Convert to tensor and apply transformations
    img_tensor = _transform(img).unsqueeze(0).to(_device)

    # Model inference
    with torch.no_grad():
        outputs = _model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        # Debug: log top-3 predictions
        try:
            topk_conf, topk_idx = torch.topk(
                probabilities, k=min(3, probabilities.size(1)), dim=1)
            top_items = []
            for rank in range(topk_idx.size(1)):
                cls_idx = topk_idx[0, rank].item()
                writer_one_based = cls_idx + 1
                writer_name = _id_to_writer_runtime.get(
                    cls_idx, f"id{writer_one_based}")
                top_items.append(
                    f"{writer_name}({writer_one_based})={topk_conf[0, rank].item():.3f}")
            logger.info("Top-3: " + ", ".join(top_items))
        except Exception as _e:
            logger.debug(f"Top-k logging skipped: {_e}")

        confidence, predicted_id = torch.max(probabilities, 1)
        writer_id = predicted_id.item() + 1
        confidence_score = confidence.item()

        writer_display = _id_to_writer_runtime.get(
            writer_id - 1, f"Writer {writer_id}")
        logger.info(
            f"Model prediction: writer_id={writer_id} ({writer_display}), confidence={confidence_score:.3f}")

        threshold = MEDIUM_CONFIDENCE_THRESHOLD
        if confidence_score >= 0.9:
            conf_level = "high"
        elif confidence_score >= threshold:
            conf_level = "medium"
        else:
            conf_level = "low"

        result = ClassificationResult(
            slip_id=slip_id,
            writer_id=writer_id,
            confidence=confidence_score,
            confidence_level=conf_level,
        )

    return result.model_dump(by_alias=True)


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    num_writers = len(_class_order) if _class_order else len(ALL_WRITERS)
    writer_ids = list(range(1, num_writers + 1))
    threshold = MEDIUM_CONFIDENCE_THRESHOLD

    # Get model type
    if isinstance(_model, EfficientNetClassifier):
        model_type = "EfficientNetClassifier"
    else:
        model_type = "Unknown"

    return {
        "model_type": model_type,
        "num_writers": num_writers,
        "available_writers": writer_ids,
        "efficientnet_model": isinstance(_model, EfficientNetClassifier),
        "writer_names": {str(i + 1): _id_to_writer_runtime.get(i, f"Writer {i + 1}") for i in range(num_writers)},
        "input_size": IMAGE_SIZE,
        "device": str(_device),
        "confidence_thresholds": {
            "high": 0.9,
            "medium": threshold,
            "low": 0.0
        },
        "business_threshold": threshold  # Threshold for .NET application to tag slips
    }


@app.get("/")
async def root():
    """API root - basic info"""
    demo_info = ""
    writers_info = f"{len(ALL_WRITERS)} writers"

    # Get actual model type
    if _model is None:
        model_type = "Unknown"
    elif isinstance(_model, EfficientNetClassifier):
        model_type = "EfficientNet"
    else:
        model_type = "Unknown"

    return {
        "api": f"BetFred Writer Classification Service{demo_info}",
        "version": app.version,
        "model": model_type,
        "writers": writers_info,
        "purpose": "Enhanced handwriting classification with improved accuracy",
        "endpoints": {
            "health": "/health",
            "classify": "/classify-anonymous",
            "model_info": "/model-info"
        },
        "integration": ".NET BetFred Application"
    }

if __name__ == "__main__":
    import uvicorn

    # Configuration for local development
    config = {
        "app": "classification_api:app",
        "host": "0.0.0.0",
        "port": 8001,
        "reload": True,
        "log_level": "info"
    }

    logger.info("Starting BetFred Classification API server...")
    logger.info(f"   URL: http://localhost:8001")
    logger.info(f"   Health check: http://localhost:8001/health")
    logger.info(f"   API docs: http://localhost:8001/docs")

    uvicorn.run(**config)
