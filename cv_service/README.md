# CV Service — EfficientNet Classification

Key endpoints

- GET /health
- GET /model-info
- POST /classify-anonymous

Scripts

- Deidentify dataset:

  python -m cv_service.scripts.deidentify_slips --src-dir cv_service/slips --out-dir cv_service/slips_anon --mapping-file cv_service/slips_deid_mapping.csv

- Stratified split:

  python -m cv_service.scripts.split_dataset --data-dir cv_service/slips_anon --out-dir cv_service/dataset_splits --train 0.7 --val 0.2 --test 0.1 --seed 42

Training

- Entry: python -m cv_service.training.train_model
- Best model saved to cv_service/trained_models/best_efficientnet_classifier.pth
- Labels sidecar: cv_service/trained_models/best_efficientnet_classifier.labels.json

Serving

- Start API: python -m classification_api
- Base URL: <http://localhost:8001/>

A machine learning service to classify handwritten bet slips for safer gambling monitoring.

## Overview

This service provides a REST API for classifying handwritten bet slips to identify individual writers. It uses EfficientNet as the single model, ensuring high accuracy and a simple, robust pipeline.

## Features

- **EfficientNet Classification**: Uses EfficientNet-B0 as the model for optimal accuracy-efficiency balance
- **REST API**: Easy integration with the .NET application
- **Confidence Scoring**: Provides confidence levels for each classification
- **Preprocessing**: Matches training pipeline (resize, grayscale to 3ch, normalize)


## Models

The service uses a model hierarchy:

1. **EfficientNet-B0**: Optimized architecture balancing accuracy and computational efficiency

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- FastAPI
- OpenCV
- NumPy
- scikit-learn

### Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Train the models:

```bash
python -m cv_service.training.train_model
```

## Usage

### Start the API

```bash
python -m classification_api
```

The API will be available at `http://localhost:8001`.

## Key endpoints

- **POST /classify-anonymous**: Classify a single handwritten bet slip
- **GET /health**: Check API health status
- **GET /model-info**: Get model information

## Scripts

```python
import requests, json

url = "http://localhost:8001/classify-anonymous"
files = {"file": ("123.jpg", open("path/to/slip.jpg", "rb"))}
response = requests.post(url, files=files)
print(json.dumps(response.json(), indent=2))
```

Additional Tools

- **api_diagnostics.py**: Check system status and model availability

## Configuration

Configuration is in a single file: `cv_service/config.py`.

This includes:

- Model configuration
- Path configurations
- Performance thresholds
- Model parameters
- Image sizes
- File paths

## Training

To train the models:

```bash
python -m cv_service.training.train_model
```

This will:

1. Train the EfficientNet model
2. Evaluate its performance
3. Save the best model weights and configurations

## Integration

This service is designed to integrate with the .NET application through the ClassificationService component.

