# BetFred Handwriting Classification Service

A machine learning service to classify handwritten bet slips for safer gambling monitoring.

## Management Tools

This service includes simple management tools:

- **train.py**: Tool to train the classification models
- **run_api.py**: Launch the API server

## Overview

This service provides a REST API for classifying handwritten bet slips to identify individual writers. It uses EfficientNet as the primary model with DenseNet as fallback, ensuring high accuracy and robustness.

## Features

- **EfficientNet Classification**: Uses EfficientNet-B0 as the primary model for optimal accuracy-efficiency balance
- **REST API**: Easy integration with the BetFred .NET application
- **Confidence Scoring**: Provides confidence levels for each classification
- **Robust Preprocessing**: Handles image variations with advanced preprocessing
- **Fallback Mechanism**: Automatically falls back to DenseNet121 if the EfficientNet model is unavailable

## Models

The service uses a model hierarchy:

1. **EfficientNet-B0**: Primary model - optimized architecture balancing accuracy and computational efficiency
2. **DenseNet121**: Fallback model - dense connections for improved gradient flow
3. **ResNet18**: Legacy support - maintained for backwards compatibility

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

2. Train the models:

```bash
python train.py
```

## Usage

### Start the API

```bash
python run_api.py
```

The API will be available at `http://localhost:8001`.

### API Endpoints

- **POST /classify-anonymous**: Classify handwritten bet slips
- **GET /health**: Check API health status
- **GET /model-info**: Get model information
- **GET /demo-status**: Check demo configuration status

### Python Client Example

```python
import requests
import json

url = "http://localhost:8001/classify-anonymous"
files = [("files", ("123.jpg", open("path/to/slip.jpg", "rb")))]

response = requests.post(url, files=files)
results = response.json()

print(json.dumps(results, indent=2))
```

## Additional Tools

This service includes several additional tools:

- **api_diagnostics.py**: Tool to check system status and model availability

- **create_dummy_models.py**: Create placeholder models for testing

## Configuration

Configuration is defined in `src/utils/config.py`:

- Model parameters
- Image sizes
- Confidence thresholds
- File paths

## Training

To train the models:

```bash
python -m src.training.train_model
```

This will:

1. Train individual models (ResNet18, EfficientNet, DenseNet)
2. Evaluate their performance
3. Identify the best-performing model (EfficientNet)
4. Save all model weights and configurations
5. Set up EfficientNet as the primary model and DenseNet as fallback

## Demo Mode

The service includes a demo mode that can be enabled by creating a `demo_config.py` file at the root directory with specific writer configurations.

## Integration

This service is designed to integrate with the BetFred .NET application through the ClassificationService component.

## License

Proprietary - BetFred 2025
