import os
import numpy as np
from datetime import datetime
from pathlib import Path
import time
import faulthandler
from PIL import Image
import joblib
import requests
from dotenv import load_dotenv
from typing import Any, Optional, Union, cast

load_dotenv()

# -------------------
# Config
# -------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models/aptos_classifier.pkl"
SCALER_PATH = BASE_DIR / "models/scaler.pkl"
WEIGHTS_PATH = BASE_DIR / "models/checkpoints/retfound_cfp_vit_large_clean.pth"

CONFIDENCE_THRESHOLD = 0.60

# Remote inference (recommended for Render Free tier)
RETFOUND_REMOTE_URL = os.getenv("RETFOUND_REMOTE_URL", "").strip().rstrip("/")
RETFOUND_REMOTE_TOKEN = os.getenv("RETFOUND_REMOTE_TOKEN", "").strip()
RETFOUND_REMOTE_TIMEOUT_S = float(os.getenv("RETFOUND_REMOTE_TIMEOUT_S", "60"))
RETFOUND_HF_REPO_ID = os.getenv("RETFOUND_HF_REPO_ID", "gadbertrand/retfound-eye-diagnosis").strip()
RETFOUND_HF_FILENAME = os.getenv("RETFOUND_HF_FILENAME", "retfound_cfp_vit_large_clean.pth").strip()

# Render free tier typically has 512MB RAM; local RETFound is likely to OOM.
IS_RENDER = bool(os.getenv("RENDER")) or bool(os.getenv("RENDER_SERVICE_ID"))
ALLOW_LOCAL_RETF0UND_ON_RENDER = os.getenv("ALLOW_LOCAL_RETFOUND_ON_RENDER", "0").lower() in {
    "1",
    "true",
    "yes",
}

# -------------------
# Clinical mappings
# -------------------
severity_map = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

clinical_description = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Microaneurysms only. Early signs present.",
    2: "Moderate NPDR with exudates or cotton-wool spots.",
    3: "Severe NPDR with hemorrhages or IRMA.",
    4: "Proliferative DR with neovascularization. High risk of vision loss."
}

recommendations = {
    0: ["Annual eye exam.", "Maintain glucose & BP control."],
    1: ["Follow-up in 12 months.", "Optimize diabetes management."],
    2: ["Follow-up in 6 months.", "Refer to ophthalmologist."],
    3: ["Urgent referral (1–2 months).", "Consider PRP treatment."],
    4: ["IMMEDIATE referral.", "Anti-VEGF or surgery may be needed."]
}

risk_level = {
    0: "LOW",
    1: "LOW-MODERATE",
    2: "MODERATE",
    3: "HIGH",
    4: "CRITICAL"
}

followup_timeline = {
    0: "12 months",
    1: "12 months",
    2: "6 months",
    3: "1–2 months",
    4: "Immediate"
}

referral_required = {
    0: False,
    1: False,
    2: True,
    3: True,
    4: True
}

# -------------------
# Lazy-loaded globals
# -------------------
clf: Any = None
scaler: Any = None
# Either the string literal "remote" or a local RETFoundExtractor instance.
extractor: Optional[Union[str, Any]] = None


def _log(msg: str):
    # Render can buffer stdout; force flush so logs appear immediately.
    print(msg, flush=True)


def _extract_features_remote(pil_image: Image.Image) -> np.ndarray:
    if not RETFOUND_REMOTE_URL:
        raise RuntimeError(
            "RETFOUND_REMOTE_URL is not set. On Render Free tier, RETFound must be offloaded."
        )

    headers = {}
    if RETFOUND_REMOTE_TOKEN:
        headers["Authorization"] = f"Bearer {RETFOUND_REMOTE_TOKEN}"

    import io

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    url = f"{RETFOUND_REMOTE_URL}/extract"
    _log(f"🌐 Calling remote feature extractor: {url}")
    resp = requests.post(
        url,
        headers=headers,
        files={"file": ("image.jpg", buf, "image/jpeg")},
        timeout=RETFOUND_REMOTE_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    feats = data.get("features")
    if not isinstance(feats, list) or not feats:
        raise RuntimeError(f"Remote extractor returned invalid payload keys={list(data.keys())}")
    return np.asarray(feats, dtype=np.float32)


def _load_local_extractor():
    from src.models.retfound import RETFoundLoader
    from src.utils.feature_extractor import RETFoundExtractor

    if WEIGHTS_PATH.exists():
        _log(f"📁 Using local RETFound weights: {WEIGHTS_PATH}")
        loader = RETFoundLoader(weights_path=str(WEIGHTS_PATH), device="cpu")
    elif RETFOUND_HF_REPO_ID and RETFOUND_HF_FILENAME:
        _log(
            "⬇️ Local RETFound weights not found; downloading from "
            f"{RETFOUND_HF_REPO_ID}/{RETFOUND_HF_FILENAME}"
        )
        loader = RETFoundLoader(
            repo_id=RETFOUND_HF_REPO_ID,
            filename=RETFOUND_HF_FILENAME,
            device="cpu",
        )
    else:
        raise RuntimeError(
            "Local RETFound fallback is enabled, but no checkpoint is available. "
            "Set RETFOUND_REMOTE_URL, place weights at "
            f"{WEIGHTS_PATH}, or configure RETFOUND_HF_REPO_ID and "
            "RETFOUND_HF_FILENAME."
        )

    return RETFoundExtractor(loader.load())


def _extract_features_local(pil_image: Image.Image) -> np.ndarray:
    if extractor is None or extractor == "remote":
        raise RuntimeError("Local RETFound extractor is not initialized.")
    local_extractor = cast(Any, extractor)
    return np.asarray(local_extractor.extract(pil_image), dtype=np.float32)


def _extract_features(pil_image: Image.Image) -> np.ndarray:
    if extractor == "remote":
        return _extract_features_remote(pil_image)
    return _extract_features_local(pil_image)


def _init_retfound_extractor():
    if RETFOUND_REMOTE_URL:
        _log(f"✅ Using remote RETFound feature extractor: {RETFOUND_REMOTE_URL}")
        return "remote"

    if IS_RENDER and not ALLOW_LOCAL_RETF0UND_ON_RENDER:
        raise RuntimeError(
            "RETFOUND_REMOTE_URL is not set. On Render Free tier (512MB), "
            "RETFound must be offloaded to a remote extractor service. "
            "Set RETFOUND_REMOTE_URL (and optional RETFOUND_REMOTE_TOKEN), "
            "or set ALLOW_LOCAL_RETFOUND_ON_RENDER=1 if you know your instance has enough RAM."
        )

    _log("🧠 RETFOUND_REMOTE_URL not set; falling back to local RETFound extractor")
    t_ret = time.time()
    local = _load_local_extractor()
    _log(f"✅ Local RETFound extractor loaded in {time.time() - t_ret:.2f}s")
    return local


def load_models():
    """
    Load models only once (lazy loading).
    On Render, keep startup fast by loading on first request.
    """
    global clf, scaler, extractor

    if clf is None:
        t0 = time.time()
        _log("🚀 Loading models...")

        # Avoid OpenMP/BLAS oversubscription/hangs on small instances.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        # If something hangs, dump stack traces to logs periodically.
        # (Helps pinpoint whether we're stuck importing sklearn, reading files, etc.)
        try:
            faulthandler.enable(all_threads=True)
            faulthandler.dump_traceback_later(120, repeat=True)
        except Exception:
            pass

        _log("🐍 Importing ML modules...")
        t_imp = time.time()
        # Import only what we need for sklearn inference.
        _log(f"🐍 Imports completed in {time.time() - t_imp:.2f}s")

        # Load ML components
        try:
            clf_size = MODEL_PATH.stat().st_size
        except Exception:
            clf_size = None
        _log(
            f"📦 Loading classifier from {MODEL_PATH}"
            + (f" ({clf_size/1024/1024:.1f} MB)" if clf_size is not None else "")
        )
        _log("📦 Importing scikit-learn (needed for unpickling)...")
        t_skl = time.time()
        import sklearn  # noqa: F401
        _log(f"✅ scikit-learn import done in {time.time() - t_skl:.2f}s")

        def _patch_classifier(obj: Any) -> Any:
            # Patch minimal attributes for runtime safety across sklearn pickle versions.
            if hasattr(obj, "__class__") and obj.__class__.__name__ == "LogisticRegression":
                if not hasattr(obj, "multi_class"):
                    setattr(obj, "multi_class", "multinomial")
            return obj

        t_clf = time.time()
        clf = _patch_classifier(joblib.load(MODEL_PATH))
        _log(f"✅ Classifier loaded in {time.time() - t_clf:.2f}s")

        try:
            scaler_size = SCALER_PATH.stat().st_size
        except Exception:
            scaler_size = None
        _log(
            f"📦 Loading scaler from {SCALER_PATH}"
            + (f" ({scaler_size/1024/1024:.1f} MB)" if scaler_size is not None else "")
        )
        t_scaler = time.time()
        scaler = joblib.load(SCALER_PATH)
        _log(f"✅ Scaler loaded in {time.time() - t_scaler:.2f}s")

        extractor = _init_retfound_extractor()

        _log(f"✅ Models loaded successfully in {time.time() - t0:.2f}s")


# -------------------
# Prediction function
# -------------------
def run_prediction(image_path: str):
    try:
        t0 = time.time()
        # Ensure models are loaded
        load_models()

        # Load image
        _log("🖼️ Reading input image...")
        img = Image.open(image_path).convert("RGB")

        # Feature extraction
        _log("🧬 Extracting features...")
        t_feat = time.time()
        features = _extract_features(img)
        _log(f"🧬 Feature extraction done in {time.time() - t_feat:.2f}s")
        features = np.array(features).flatten().reshape(1, -1)
        features_scaled = cast(Any, scaler).transform(features)

        # Prediction
        _log("🔮 Running classifier prediction...")
        pred = int(cast(Any, clf).predict(features_scaled)[0])
        proba = cast(Any, clf).predict_proba(features_scaled)[0]
        confidence = float(proba[pred])

        # Ranked probabilities
        ranked_probs = sorted(
            [
                {
                    "stage": i,
                    "label": severity_map[i],
                    "probability": float(round(proba[i], 4)),
                    "is_prediction": i == pred
                }
                for i in range(len(proba))
            ],
            key=lambda x: x["probability"],
            reverse=True
        )

        differential = next(
            (r for r in ranked_probs if r["stage"] != pred), None
        )
        referral_urgency = "NOT REQUIRED"
        if pred == 4:
            referral_urgency = "IMMEDIATE"
        elif pred == 3:
            referral_urgency = "URGENT"
        elif pred == 2:
            referral_urgency = "ROUTINE"

        return {
            "image": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": round(time.time() - t0, 3),

            "diagnosis": {
                "predicted_stage": pred,
                "predicted_label": severity_map[pred],
                "risk_level": risk_level[pred],
                "clinical_summary": clinical_description[pred],
            },

            "confidence": {
                "score": round(confidence, 4),
                "percentage": round(confidence * 100, 1),
                "low_confidence_flag": confidence < CONFIDENCE_THRESHOLD,
                "interpretation": (
                    "Low confidence — manual review recommended."
                    if confidence < CONFIDENCE_THRESHOLD
                    else "Confident prediction."
                )
            },

            "probabilities": {
                "by_stage": {
                    severity_map[i]: float(round(proba[i], 4))
                    for i in range(len(proba))
                },
                "ranked": ranked_probs,
                "differential_diagnosis": (
                    {
                        "stage": differential["stage"],
                        "label": differential["label"],
                        "probability": differential["probability"]
                    } if differential else None
                )
            },

            "clinical_guidance": {
                "recommendations": recommendations[pred],
                "followup_timeline": followup_timeline[pred],
                "referral_required": referral_required[pred],
                "referral_urgency": referral_urgency
            },

            "alerts": {
                "critical": pred == 4,
                "urgent": pred == 3,
                "low_confidence": confidence < CONFIDENCE_THRESHOLD,
                "requires_immediate_action": pred >= 3
            }
        }

    except Exception as e:
        _log(f"❌ Prediction error: {e}")
        return {
            "error": str(e),
            "image": os.path.basename(str(image_path)),
            "timestamp": datetime.now().isoformat(),
        }


# -------------------
# CLI support
# -------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        result = run_prediction(sys.argv[1])
        print(json.dumps(result, indent=2))