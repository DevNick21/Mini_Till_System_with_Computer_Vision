"""
Classification API for Writer Identification System
Provides REST endpoints for handwriting classification with confidence scoring
"""
from core import init_model, classify_image, model_info
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from pathlib import Path

current_dir = Path(__file__).parent


# Configure logging without colors
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',  # Simple format without colors
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Writer Classification API (EfficientNet)",
    version="1.0.0",
    description="Enhanced handwriting classification service using EfficientNet architecture"
)

# CORS for local frontend and backend enabled
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5113",
        "http://127.0.0.1:5113",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def to_camel(s: str) -> str:
    """Convert snake_case to camelCase for JSON field names"""
    parts = s.split('_')
    return parts[0] + ''.join(p.title() for p in parts[1:])


class ClassificationResult(BaseModel):
    # Use camelCase aliases for JSON
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    writer_id: int          # Numeric writer ID (1-13)
    confidence: float


def initialize_model():
    try:
        init_model()
        logger.info(
            "Classification model initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting Classification API...")
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
    try:
        info = model_info()
        device = info.get("device", "unknown")
        writers = info.get("num_writers", 0)
        loaded = info.get("model_type", "Unknown") != "Unknown"
    except Exception:
        device = "unknown"
        writers = 0
        loaded = False
    return {
        "status": "healthy" if loaded else "degraded",
        "api": "Writer Classification",
        "model_loaded": loaded,
        "device": device,
        "writers": writers,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/classify-anonymous", response_model=ClassificationResult)
async def classify_handwriting(file: UploadFile = File(...)):
    """Classify a single handwriting slip and return the result"""

    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    logger.info(f"Processing slip image for classification: {file.filename}")

    # Extract slip ID
    if file.filename and '.' in file.filename:
        slip_id_str = file.filename.split('.')[0]
        try:
            slip_id = int(slip_id_str)
            logger.info(f"Extracted SlipId: {slip_id}")
        except ValueError:
            logger.warning(
                f"Could not parse slip ID from filename: {file.filename}")

    # Read bytes once and delegate to core inference
    image_data = await file.read()
    try:
        writer_id, confidence_score, _conf_level = classify_image(image_data)
        result = ClassificationResult(
            writer_id=writer_id,
            confidence=confidence_score,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result.model_dump(by_alias=True)


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    return model_info()


@app.get("/")
async def root():
    """API root - basic info"""
    demo_info = ""
    try:
        info = model_info()
        writers_info = f"{info.get('num_writers', 0)} writers"
    except Exception:
        writers_info = "unknown writers"

    # Pull model type via core
    info_obj = model_info()
    model_type = info_obj.get("model_type", "Unknown")

    return {
        "api": f"Writer Classification Service{demo_info}",
        "version": app.version,
        "model": model_type,
        "writers": writers_info,
        "purpose": "Handwriting classification",
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

    logger.info("Starting Classification API server...")
    logger.info(f"   URL: http://localhost:8001")
    logger.info(f"   Health check: http://localhost:8001/health")
    logger.info(f"   API docs: http://localhost:8001/docs")

    uvicorn.run(**config)
