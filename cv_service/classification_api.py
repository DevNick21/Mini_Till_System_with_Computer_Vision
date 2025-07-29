"""
Classification API for BetFred Writer Identification System
Provides REST endpoints for handwriting classification with confidence scoring
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
import logging
from datetime import datetime
from pydantic import BaseModel
import os
import sys
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

# Default writer display names
WRITER_DISPLAY_NAMES = {i: f"Writer {i}" for i in range(1, 14)}

# Import models and config
try:
    # Import models and utils with improved structure
    from src.models import EfficientNetClassifier, DenseNetClassifier
    from src.utils import *
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
    version="2.1.0",
    description="Enhanced handwriting classification service using EfficientNet architecture"
)


class ClassificationResult(BaseModel):
    slip_id: int
    writer_id: int          # Numeric writer ID (1-13)
    confidence: float
    confidence_level: str   # "high", "medium", "low"


class ClassificationResponse(BaseModel):
    results: List[ClassificationResult]
    summary: Dict[str, Any]
    timestamp: str


# Global model variables
_model = None
_device = None
_transform = None


def initialize_model():
    """Initialize the trained handwriting classification model (EfficientNet)"""
    global _model, _device, _transform

    try:
        # Determine device
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {_device}")

        # Check if the EfficientNet model is available
        efficientnet_path = os.path.join(
            MODEL_SAVE_PATH, "best_efficientnet_classifier.pth")

        # Try to load EfficientNet model first
        if os.path.exists(efficientnet_path):
            try:
                logger.info("EfficientNet model found, using as primary model")
                _model = EfficientNetClassifier(num_writers=len(ALL_WRITERS))
                _model.load_state_dict(torch.load(
                    efficientnet_path, map_location=_device))
                _model.to(_device)
                _model.eval()
                model_path = efficientnet_path
                logger.info("EfficientNet model loaded successfully")
            except Exception as e:
                logger.error(f"✗ Error loading EfficientNet model: {e}")
                _model = None
        else:
            logger.warning(
                "⚠️  EfficientNet model not found, checking for legacy model")
            _model = None

        # If EfficientNet initialization failed, fall back to the DenseNet model
        if _model is None:
            logger.info("Loading DenseNet model as fallback...")
            densenet_path = os.path.join(
                MODEL_SAVE_PATH, "best_densenet_classifier.pth")

            if os.path.exists(densenet_path):
                try:
                    # Try to load DenseNet model
                    _model = DenseNetClassifier(num_writers=len(ALL_WRITERS))
                    _model.load_state_dict(torch.load(
                        densenet_path, map_location=_device))
                    _model.to(_device)
                    _model.eval()
                    model_path = densenet_path
                    logger.info(
                        "DenseNet model loaded successfully as fallback")
                except Exception as e:
                    logger.error(f"✗ Error loading DenseNet model: {e}")
                    raise
            else:
                raise FileNotFoundError(
                    f"No models found in {MODEL_SAVE_PATH}")

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
        logger.info(f"   Writers: 13 (IDs 1-13)")

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
        "writers": 13,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/classify-anonymous", response_model=ClassificationResponse)
async def classify_handwriting(files: List[UploadFile] = File(...)):
    """
    Classify handwriting samples and return anonymous writer IDs
    """

    if _model is None:
        raise HTTPException(
            status_code=503, detail="Classification model not loaded")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    logger.info(f"Processing {len(files)} slip images for classification")

    results = []
    processing_errors = []

    for file_idx, file in enumerate(files):
        try:
            # FIX: Better filename parsing
            logger.info(f"Processing file: {file.filename}")

            # Extract slip ID from filename (format: "123.jpg")
            if file.filename and '.' in file.filename:
                slip_id_str = file.filename.split('.')[0]
                try:
                    slip_id = int(slip_id_str)
                    logger.info(f"Extracted SlipId: {slip_id}")
                except ValueError:
                    logger.error(
                        f"Could not parse slip ID from filename: {file.filename}")
                    slip_id = file_idx + 1  # Fallback to index-based ID
            else:
                logger.error(f"Invalid filename format: {file.filename}")
                slip_id = file_idx + 1  # Fallback to index-based ID

            # Read and decode image
            image_data = await file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError(f"Could not decode image {file.filename}")

            logger.info(f"Image decoded successfully: {img.shape}")

            # Preprocess image (same as training pipeline)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

            # Convert to tensor and apply transformations
            img_tensor = _transform(img).unsqueeze(0).to(_device)

            # Model inference
            with torch.no_grad():
                outputs = _model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)

                # If in demo mode, only consider the specified writers
                if DEMO_MODE:
                    # Create a mask for demo writers (convert from 1-based to 0-based indices)
                    demo_indices = [writer_id -
                                    1 for writer_id in DEMO_WRITERS]

                    # Extract probabilities only for demo writers
                    demo_probs = torch.zeros_like(probabilities)
                    for idx in demo_indices:
                        if idx >= 0 and idx < probabilities.size(1):
                            demo_probs[0, idx] = probabilities[0, idx]

                    # Get max probability from demo writers only
                    confidence, predicted_id = torch.max(demo_probs, 1)
                else:
                    # Standard behavior - consider all writers
                    confidence, predicted_id = torch.max(probabilities, 1)

                # FIX: Ensure writer ID is 1-based (model outputs 0-12, we want 1-13)
                writer_id = predicted_id.item() + 1
                confidence_score = confidence.item()

                # Get display name if available
                writer_display = WRITER_DISPLAY_NAMES.get(
                    writer_id, f"Writer {writer_id}")

                logger.info(
                    f"Model prediction: writer_id={writer_id} ({writer_display}), confidence={confidence_score:.3f}")

                # In demo mode, use a potentially lower confidence threshold
                threshold = DEMO_CONFIDENCE_BASE if DEMO_MODE else 0.75

                # Determine confidence level for business logic
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
                    confidence_level=conf_level
                )

                results.append(result)

                logger.info(
                    f"Classified Slip #{slip_id}: Writer {writer_id} ({confidence_score:.3f}, {conf_level})")

        except ValueError as e:
            error_msg = f"Invalid filename format: {file.filename} - {str(e)}"
            processing_errors.append(error_msg)
            logger.warning(error_msg)

        except Exception as e:
            error_msg = f"Error processing {file.filename}: {str(e)}"
            processing_errors.append(error_msg)
            logger.error(error_msg)

    # ... rest of the method stays the same
    if not results and processing_errors:
        # All files failed to process
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process any files. Errors: {'; '.join(processing_errors)}"
        )

    # Calculate summary statistics
    high_conf_count = len([r for r in results if r.confidence_level == "high"])
    medium_conf_count = len(
        [r for r in results if r.confidence_level == "medium"])
    low_conf_count = len([r for r in results if r.confidence_level == "low"])
    threshold_eligible = len([r for r in results if r.confidence >= 0.75])

    response = ClassificationResponse(
        results=results,
        summary={
            "total_processed": len(results),
            "high_confidence": high_conf_count,
            "medium_confidence": medium_conf_count,
            "low_confidence": low_conf_count,
            "threshold_eligible": threshold_eligible,
            "processing_errors": len(processing_errors),
            "average_confidence": sum(r.confidence for r in results) / len(results) if results else 0,
            "unique_writers": len(set(r.writer_id for r in results))
        },
        timestamp=datetime.utcnow().isoformat()
    )

    logger.info(
        f"Classification complete: {len(results)} classified, {threshold_eligible} eligible for thresholds")

    if processing_errors:
        logger.warning(
            f"⚠️  {len(processing_errors)} files had processing errors")

    return response


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # If in demo mode, only report the demo writers
    writer_ids = DEMO_WRITERS if DEMO_MODE else list(range(1, 14))
    threshold = DEMO_CONFIDENCE_BASE if DEMO_MODE else 0.75

    # Get model type
    if isinstance(_model, EfficientNetClassifier):
        model_type = "EfficientNetClassifier"
    elif isinstance(_model, DenseNetClassifier):
        model_type = "DenseNetClassifier"
    else:
        model_type = "ResNetClassifier"

    return {
        "model_type": model_type,
        "num_writers": 13,
        "available_writers": writer_ids,
        "demo_mode": DEMO_MODE,
        "efficientnet_model": isinstance(_model, EfficientNetClassifier),
        "writer_names": {str(k): v for k, v in WRITER_DISPLAY_NAMES.items() if k in writer_ids},
        "input_size": IMAGE_SIZE,
        "device": str(_device),
        "confidence_thresholds": {
            "high": 0.9,
            "medium": threshold,
            "low": 0.0
        },
        "business_threshold": threshold  # Threshold for .NET application to tag slips
    }


@app.get("/demo-status")
async def get_demo_status():
    """Get information about the demo configuration"""
    return {
        "demo_mode": DEMO_MODE,
        "available_writers": DEMO_WRITERS,
        "writer_names": {str(k): v for k, v in WRITER_DISPLAY_NAMES.items() if k in DEMO_WRITERS},
        "confidence_threshold": DEMO_CONFIDENCE_BASE,
        "sample_bet_slips": [f"slips/{WRITER_DISPLAY_NAMES[wid].lower().replace(' ', '_')}.jpg" for wid in DEMO_WRITERS if wid in WRITER_DISPLAY_NAMES],
        "instructions": "Use these sample bet slips for your demo to ensure correct classification."
    }


@app.get("/")
async def root():
    """API root - basic info"""
    demo_info = " (DEMO MODE)" if DEMO_MODE else ""
    writers_info = f"{len(DEMO_WRITERS)} writers" if DEMO_MODE else "13 writers"

    # Get actual model type
    if _model is None:
        model_type = "Unknown"
    elif isinstance(_model, EfficientNetClassifier):
        model_type = "EfficientNet"
    elif isinstance(_model, DenseNetClassifier):
        model_type = "DenseNet"
    else:
        model_type = "ResNet"

    return {
        "api": f"BetFred Writer Classification Service{demo_info}",
        "version": "2.0.0",
        "model": model_type,
        "writers": writers_info,
        "purpose": "Enhanced handwriting classification with improved accuracy",
        "endpoints": {
            "health": "/health",
            "classify": "/classify-anonymous",
            "model_info": "/model-info",
            "demo_status": "/demo-status"
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
