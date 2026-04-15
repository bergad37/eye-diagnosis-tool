import os
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
import joblib

from src.utils.feature_extractor import RETFoundExtractor
from src.models.retfound import RETFoundLoader

# -------------------
# Config
# -------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models/aptos_classifier.pkl"
SCALER_PATH = BASE_DIR / "models/scaler.pkl"
WEIGHTS_PATH = BASE_DIR / "models/checkpoints/retfound_cfp_vit_large_clean.pth"

CONFIDENCE_THRESHOLD = 0.60

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
clf = None
scaler = None
extractor = None


def load_models():
    """
    Load models only once (lazy loading).
    On Render, keep startup fast by loading on first request.
    """
    global clf, scaler, extractor

    if clf is None:
        print("🚀 Loading models...")

        # Load ML components
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Ensure Hugging Face cache is writable on Render.
        # Render filesystems are ephemeral; /tmp is a safe writable location.
        os.environ.setdefault("HF_HOME", "/tmp/huggingface")
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/huggingface/hub")

        # Load RETFound model (prefer local weights if present; otherwise download)
        if WEIGHTS_PATH.exists():
            loader = RETFoundLoader(
                weights_path=str(WEIGHTS_PATH),
                device="cpu",
            )
        else:
            loader = RETFoundLoader(
                repo_id="gadbertrand/retfound-eye-diagnosis",
                filename="retfound_cfp_vit_large_clean.pth",
                device="cpu",
            )

        retfound_model = loader.load()
        extractor = RETFoundExtractor(retfound_model)

        print("✅ Models loaded successfully")


# -------------------
# Prediction function
# -------------------
def run_prediction(image_path: str):
    try:
        # Ensure models are loaded
        load_models()

        # Load image
        img = Image.open(image_path).convert("RGB")

        # Feature extraction
        features = extractor.extract(np.array(img))
        features = np.array(features).flatten().reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Prediction
        pred = int(clf.predict(features_scaled)[0])
        proba = clf.predict_proba(features_scaled)[0]
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

        return {
            "image": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat(),

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
                "referral_urgency": (
                    "IMMEDIATE" if pred == 4
                    else "URGENT" if pred == 3
                    else "ROUTINE" if pred == 2
                    else "NOT REQUIRED"
                )
            },

            "alerts": {
                "critical": pred == 4,
                "urgent": pred == 3,
                "low_confidence": confidence < CONFIDENCE_THRESHOLD,
                "requires_immediate_action": pred >= 3
            }
        }

    except Exception as e:
        return {
            "error": str(e),
            "image": os.path.basename(str(image_path)),
            "timestamp": datetime.now().isoformat()
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