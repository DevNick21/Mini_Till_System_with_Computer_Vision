from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import re
from typing import Optional, Tuple

app = FastAPI(title="Slip OCR Service", version="0.1.0",
              description="Extract fields (stake, etc.) from slip images")

# CORS for local dev (React on 3000, .NET on 5113)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5113",
    "http://127.0.0.1:5113",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ocr_reader = None


def _init_reader(lang: str = 'en'):
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader([lang], gpu=False, verbose=False)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR: {e}")


def _preprocess(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    white_ratio = (th == 255).mean()
    if white_ratio < 0.5:
        th = 255 - th
    return th


stake_pattern = re.compile(
    r"(stake|stk|amount)[^\d]{0,10}([£$€₦N]?[\s]*\d+(?:[.,]\d{1,2})?)", re.IGNORECASE)


def _parse_stake(text: str) -> Optional[Tuple[float, str, Optional[str]]]:
    t = text.replace(',', '').replace('=', ':')
    currency = None
    for c, tag in [("£", "GBP"), ("$", "USD"), ("€", "EUR"), ("₦", "NGN"), ("N", "NGN")]:
        if c in t:
            currency = tag
            t = t.replace(c, '')
    m = stake_pattern.search(t)
    if m:
        val = m.group(2)
        val = val.replace(' ', '')
        # strip any leftover currency letters
        val = re.sub(r"[A-Za-z]", "", val)
        val = val.replace(',', '')
        try:
            return float(val), 'regex', currency
        except Exception:
            return None
    # fallback: find large numbers
    m2 = re.search(r"\b(\d{2,})(?:\.\d{1,2})?\b", t)
    if m2:
        try:
            return float(m2.group(1)), 'fallback', currency
        except Exception:
            return None
    return None


@app.get("/")
async def root():
    return {
        "service": "Slip OCR Service",
        "version": app.version,
        "endpoints": {"health": "/health", "stake": "/ocr/stake"}
    }


@app.get("/health")
async def health():
    try:
        _init_reader('en')
        return {"status": "ok", "reader": "ready"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/ocr/stake")
async def ocr_stake(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        _init_reader('en')
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    proc = _preprocess(img)
    # OCR
    try:
        lines = _ocr_reader.readtext(proc, detail=0, paragraph=True)
    except Exception:
        lines = _ocr_reader.readtext(proc, detail=0)

    # combine to text
    joined = "\n".join(
        [ln if isinstance(ln, str) else " ".join(map(str, ln)) for ln in lines])
    parsed = _parse_stake(joined)
    if not parsed:
        return {"stake": None, "method": None, "currency": None}
    value, method, currency = parsed
    return {"stake": value, "method": method, "currency": currency}
