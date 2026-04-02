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
MODEL_PATH = "models/aptos_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"

severity_map = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

clinical_description = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Microaneurysms only. Early signs of diabetic retinopathy present.",
    2: "More than just microaneurysms but less than severe NPDR. "
       "Hard exudates, cotton-wool spots, or venous beading may be present.",
    3: "Severe NPDR: extensive intraretinal hemorrhages, venous beading, "
       "or intraretinal microvascular abnormalities (IRMA).",
    4: "Proliferative DR: neovascularization or vitreous/pre-retinal hemorrhage present. "
       "High risk of vision loss."
}

recommendations = {
    0: [
        "Routine annual diabetic eye exam recommended.",
        "Maintain good glycemic and blood pressure control.",
        "No immediate ophthalmic intervention required."
    ],
    1: [
        "Follow-up eye exam in 12 months.",
        "Optimize blood glucose (HbA1c < 7%) and blood pressure control.",
        "Patient education on diabetes management and eye health."
    ],
    2: [
        "Follow-up eye exam in 6 months.",
        "Referral to ophthalmologist for evaluation.",
        "Strict glycemic and blood pressure management.",
        "Consider fluorescein angiography if clinically indicated."
    ],
    3: [
        "Urgent ophthalmology referral within 1–2 months.",
        "Pan-retinal photocoagulation (PRP) may be indicated.",
        "Intensive glycemic and systemic risk factor management.",
        "Regular monitoring for progression to PDR."
    ],
    4: [
        "URGENT: Immediate ophthalmology referral required.",
        "Pan-retinal photocoagulation (PRP) or anti-VEGF therapy likely needed.",
        "Vitrectomy may be required if vitreous hemorrhage is present.",
        "Strict systemic control critical to prevent further progression.",
        "Patient counseling on risk of severe vision loss."
    ]
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
    3: "1–2 months (urgent)",
    4: "Immediate"
}

referral_required = {
    0: False,
    1: False,
    2: True,
    3: True,
    4: True
}

CONFIDENCE_THRESHOLD = 0.60

# -------------------
# LOAD MODELS ONCE
# -------------------
print("🚀 Loading models...")

clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

weights = Path("models/checkpoints/retfound_cfp_vit_large_clean.pth")
loader = RETFoundLoader(weights, device="cpu")
retfound_model = loader.load()
extractor = RETFoundExtractor(retfound_model)

print("✅ Models loaded successfully")


# -------------------
# MAIN FUNCTION (API READY)
# -------------------
def run_prediction(image_path):
    try:
        # Load and process image
        img = Image.open(image_path).convert("RGB")
        features = extractor.extract(np.array(img))
        features = np.array(features).flatten().reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Predict
        pred = int(clf.predict(features_scaled)[0])
        proba = clf.predict_proba(features_scaled)[0]
        confidence = float(proba[pred])

        # Build ranked probability list (highest → lowest)
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

        # Second-highest probability stage (differential)
        differential = next((r for r in ranked_probs if r["stage"] != pred), None)

        return {
            # --- Core prediction ---
            "image": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat(),

            # --- Diagnosis ---
            "diagnosis": {
                "predicted_stage": pred,
                "predicted_label": severity_map[pred],
                "risk_level": risk_level[pred],
                "clinical_summary": clinical_description[pred],
            },

            # --- Confidence & uncertainty ---
            "confidence": {
                "score": round(confidence, 4),
                "percentage": round(confidence * 100, 1),
                "low_confidence_flag": confidence < CONFIDENCE_THRESHOLD,
                "interpretation": (
                    "Low confidence — interpret with caution and consider manual review."
                    if confidence < CONFIDENCE_THRESHOLD
                    else "Confident prediction."
                )
            },

            # --- Probability breakdown ---
            "probabilities": {
                "by_stage": {severity_map[i]: float(round(proba[i], 4)) for i in range(len(proba))},
                "ranked": ranked_probs,
                "differential_diagnosis": {
                    "stage": differential["stage"],
                    "label": differential["label"],
                    "probability": differential["probability"]
                } if differential else None
            },

            # --- Clinical guidance ---
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

            # --- Alert flags (useful for API consumers to trigger UI alerts) ---
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
# OPTIONAL CLI SUPPORT
# -------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        result = run_prediction(sys.argv[1])
        print(json.dumps(result, indent=2))