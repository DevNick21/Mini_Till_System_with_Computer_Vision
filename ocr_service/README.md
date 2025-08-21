# Slip OCR Service

A small FastAPI microservice that OCRs bet slips and extracts the stake (and future fields).

- Language: Python 3.11
- Port: 8002
- CORS: Allows localhost:3000 and 5113 for local dev

## Endpoints

- GET /health: sanity check
- POST /ocr/stake: multipart file upload (field name `file`), returns `{ stake, method, currency }`

## Run locally

```bash
# from repo root
python -m venv .venv-ocr && . .venv-ocr/Scripts/activate
pip install -r ocr_service/requirements.txt
python -m uvicorn ocr_service.ocr_api:app --host 0.0.0.0 --port 8002 --reload
```

## Notes

- Uses EasyOCR (CPU) and OpenCV headless.
- Keep this service independent from the classification service for clean boundaries.
