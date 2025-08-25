# cv_service — Writer Classification (EfficientNet-B0)

A lightweight microservice for classifying handwriting slips to anonymous writer IDs. It uses EfficientNet‑B0 with a CLAHE‑enabled preprocessing pipeline and exposes a small FastAPI.

## Structure

- `config.py` — central config (paths, thresholds, toggles like PREPROCESS_CLAHE)
- `models/efficientnet_classifier.py` — model definition (EfficientNet‑B0 backbone + classifier head)
- `core/inference.py` — model load and single‑image classification utilities
- `classification_api.py` — FastAPI app exposing health, model-info, and classify endpoints
- `training/`
  - `data_prep.py` — dataset, transforms (with CLAHE), stratified split, DataLoaders
  - `train_model.py` — training loop (AdamW, ReduceLROnPlateau, early stop), checkpointing
- `scripts/` — helpers: de-identify slips, split dataset
- `trained_models/` — saved weights and labels sidecar
- `slips/` — training data (by writer folder); or use `dataset_splits/{train,val}`

## Quick start

1. Install deps (conda or pip)

```bash
pip install -r cv_service/requirements.txt
```

2. Prepare data

- Either place images under `cv_service/slips/<writer>/...`
- Or create splits via script:

```bash
python -m cv_service.scripts.split_dataset \
  --data-dir cv_service/slips \
  --out-dir cv_service/dataset_splits \
  --train 0.7 --val 0.3 --seed 42
```

1. Train

```bash
python -m cv_service.training.train_model
```

Outputs:

- Weights: `cv_service/trained_models/best_efficientnet_classifier.pth`
- Checkpoint: `cv_service/trained_models/best_efficientnet_classifier.ckpt`
- Labels sidecar: `cv_service/trained_models/best_efficientnet_classifier.labels.json` (key: `all_writers`)

1. Serve API

```bash
python -m cv_service.classification_api
```

Base URL: <http://localhost:8001>

## Configuration

Edit `cv_service/config.py`:

- `PREPROCESS_CLAHE = True` — enables CLAHE in train/val/inference
- `IMAGE_SIZE` (default 224)
- Paths: `SLIPS_DIR`, `MODEL_SAVE_PATH`, filenames
- Training: `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `VAL_SPLIT`, patience
- Thresholds: `MEDIUM_CONFIDENCE_THRESHOLD`, `HIGH_CONFIDENCE_THRESHOLD`

## API

- GET `/health` — service status, device, model loaded
- GET `/model-info` — model type, writers, thresholds
- POST `/classify-anonymous` — form-data `file` image → `{ writerId, confidence }`

Example:

```python
import requests, json
r = requests.post(
    "http://localhost:8001/classify-anonymous",
    files={"file": ("123.jpg", open("/path/to/slip.jpg", "rb"))},
)
print(json.dumps(r.json(), indent=2))
```

## Model

- Backbone: torchvision EfficientNet‑B0 (with pretrained fallback)
- Classifier head (restored):
  - Flatten → Dropout → Linear(1280→512) → ReLU → BatchNorm → Dropout → Linear(512→256) → ReLU → BatchNorm → Dropout → Linear(256→num_writers)

## Training

- Optimizer: AdamW (weight‑decay param groups)
- Scheduler: ReduceLROnPlateau on Val Acc@1
- Early stop: configurable patience
- Metrics: Top‑1 Accuracy
- Checkpointing: saves best weights + `.ckpt` + labels sidecar

To resume (weights only), just keep the `.pth` in `trained_models/` and rerun serving.

## Notes & tips

- Old/new weights compatibility depends on classifier head. With the restored head, older checkpoints should load.
- If you change the head, bump the model filename to avoid shape mismatch.
- Keep training/inference transforms aligned (CLAHE + Resize + Grayscale(3ch) + Normalize) to avoid distribution shift.
- Batch classification has been removed by design; API is single‑image only.

## Scripts

- De‑identify slips:

```bash
python -m cv_service.scripts.deidentify_slips \
  --src-dir cv_service/slips \
  --out-dir cv_service/slips_anon \
  --mapping-file cv_service/slips_deid_mapping.csv
```

- Create dataset splits:

```bash
python -m cv_service.scripts.split_dataset \
  --data-dir cv_service/slips_anon \
  --out-dir cv_service/dataset_splits \
  --train 0.7 --val 0.3 --test 0.0 --seed 42
```

