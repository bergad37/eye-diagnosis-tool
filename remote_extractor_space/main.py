import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image


app = FastAPI()

# Hugging Face weights location (same defaults as your main app)
RETFOUND_HF_REPO_ID = os.getenv("RETFOUND_HF_REPO_ID", "gadbertrand/retfound-eye-diagnosis").strip()
RETFOUND_HF_FILENAME = os.getenv("RETFOUND_HF_FILENAME", "retfound_cfp_vit_large_clean.pth").strip()

# Only needed for private repos. huggingface_hub will pick up HF_TOKEN automatically.
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if HF_TOKEN:
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", HF_TOKEN)

_extractor = None


def _get_extractor():
    global _extractor
    if _extractor is not None:
        return _extractor

    # Import heavy deps lazily so container boots faster.
    from src.models.retfound import RETFoundLoader
    from src.utils.feature_extractor import RETFoundExtractor

    loader = RETFoundLoader(
        repo_id=RETFOUND_HF_REPO_ID,
        filename=RETFOUND_HF_FILENAME,
        device="cpu",
    )
    _extractor = RETFoundExtractor(loader.load())
    return _extractor


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected image/*, got {file.content_type}")

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")  # type: ignore[name-defined]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        extractor = _get_extractor()
        feats = extractor.extract(img)
        feats = np.asarray(feats, dtype=np.float32).flatten()
        return {"features": feats.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Small import at end (used above) to keep top minimal.
import io  # noqa: E402

