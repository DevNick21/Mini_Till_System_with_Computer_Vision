# BetFred Handwriting Classification Service

A machine learning service to classify handwritten bet slips for safer gambling monitoring.

## Management Tools

- **train.py**: Train the classification model
- **run_api.py**: Launch the API server

## Overview

This service provides a REST API for classifying handwritten bet slips to identify individual writers. It uses EfficientNet as the single model, ensuring high accuracy and a simple, robust pipeline.

## Features

- **EfficientNet Classification**: Uses EfficientNet-B0 as the model for optimal accuracy-efficiency balance
- **REST API**: Easy integration with the BetFred .NET application
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
python train.py
```

## Usage

### Start the API

```bash
python run_api.py
```

The API will be available at `http://localhost:8001`.

### API Endpoints

- **POST /classify-anonymous**: Classify a single handwritten bet slip
- **GET /health**: Check API health status
- **GET /model-info**: Get model information

### Python Client Example

```python
import requests
import json

url = "http://localhost:8001/classify-anonymous"
files = {"file": ("123.jpg", open("path/to/slip.jpg", "rb"))}

response = requests.post(url, files=files)
result = response.json()

print(json.dumps(result, indent=2))
```

## Additional Tools

This service includes an optional diagnostic:

- **api_diagnostics.py**: Check system status and model availability

## Configuration

Configuration is unified in a single file: `src/utils/config.py`.

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
python -m src.training.train_model
```

This will:

1. Train the EfficientNet model
2. Evaluate its performance
3. Save the best model weights and configurations

## Demo Mode

Removed. The service now always runs with the trained model outputs.

## Integration

This service is designed to integrate with the BetFred .NET application through the ClassificationService component.

## License

Proprietary - BetFred 2025
