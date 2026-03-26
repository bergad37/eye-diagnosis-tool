
import os
import sys
import numpy as np
import json
import csv
from datetime import datetime

try:
    import joblib
except ModuleNotFoundError:
    sys.exit(
        "joblib is not available. "
        "Make sure you run the script with the project's virtual environment "
        "(e.g. `source venv/bin/activate`) or install dependencies with "
        "`pip install -r requirements.txt`."
    )

from pathlib import Path
from PIL import Image
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.feature_extractor import RETFoundExtractor
from src.models.retfound import RETFoundLoader

# -------------------
# Config
# -------------------
MODEL_PATH = "models/aptos_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"
TEST_IMAGES_DIR = "data/aptos/test_images/"
OUTPUT_DIR = "outputs/reports/"
EXPORT_JSON = True   # Set False to skip JSON export
EXPORT_CSV  = True   # Set False to skip CSV export

# -------------------
# DR Clinical Reference Data
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

CONFIDENCE_THRESHOLD = 0.60  # Flag predictions below this as uncertain

# -------------------
# Helpers
# -------------------

def format_separator(char="─", width=60):
    return char * width

def confidence_bar(prob, width=20):
    """Simple ASCII confidence bar."""
    filled = int(prob * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {prob*100:5.1f}%"

def print_image_report(img_file, pred, proba, stage_labels):
    confidence = proba[pred]
    uncertain = confidence < CONFIDENCE_THRESHOLD

    print(format_separator())
    print(f"  Image : {img_file}")
    print(f"  Stage : {pred} — {severity_map[pred]}")
    print(f"  Risk  : {risk_level[pred]}")
    print(f"  {'⚠ LOW CONFIDENCE — interpret with caution' if uncertain else ''}")
    print()
    print("  Clinical Summary:")
    print(f"    {clinical_description[pred]}")
    print()
    print("  Probability Distribution:")
    for i, label in stage_labels.items():
        marker = " ◄" if i == pred else ""
        print(f"    {label:<20} {confidence_bar(proba[i])}{marker}")
    print()
    print("  Recommendations:")
    for rec in recommendations[pred]:
        print(f"    • {rec}")
    print()

def build_record(img_file, pred, proba):
    return {
        "image": img_file,
        "predicted_stage": int(pred),
        "predicted_label": severity_map[pred],
        "risk_level": risk_level[pred],
        "confidence": float(round(proba[pred], 4)),
        "low_confidence_flag": bool(proba[pred] < CONFIDENCE_THRESHOLD),
        "probabilities": {severity_map[i]: float(round(proba[i], 4)) for i in range(5)},
        "clinical_summary": clinical_description[pred],
        "recommendations": recommendations[pred],
        "timestamp": datetime.now().isoformat()
    }

def print_summary(records):
    print(format_separator("═"))
    print("  BATCH SUMMARY")
    print(format_separator("═"))

    preds = [r["predicted_stage"] for r in records]
    confs = [r["confidence"] for r in records]
    flagged = [r for r in records if r["low_confidence_flag"]]

    count = Counter(preds)
    print(f"  Total images processed : {len(records)}")
    print(f"  Mean confidence        : {np.mean(confs)*100:.1f}%")
    print(f"  Low-confidence flags   : {len(flagged)}")
    print()
    print("  Stage Distribution:")
    for stage in sorted(count):
        bar_len = int((count[stage] / len(records)) * 30)
        print(f"    {severity_map[stage]:<22} {'█' * bar_len} {count[stage]}")

    if flagged:
        print()
        print("  ⚠  Images Flagged for Review:")
        for r in flagged:
            print(f"    • {r['image']}  (confidence {r['confidence']*100:.1f}%)")

    critical = [r for r in records if r["predicted_stage"] == 4]
    urgent   = [r for r in records if r["predicted_stage"] == 3]
    if critical:
        print()
        print("  🚨 CRITICAL — Immediate Referral Required:")
        for r in critical:
            print(f"    • {r['image']}")
    if urgent:
        print()
        print("  ⚠  HIGH RISK — Urgent Ophthalmology Referral:")
        for r in urgent:
            print(f"    • {r['image']}")

    print(format_separator("═"))

def export_json(records, output_dir):
    path = os.path.join(output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  JSON report saved → {path}")

def export_csv(records, output_dir):
    path = os.path.join(output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    fieldnames = [
        "image", "predicted_stage", "predicted_label", "risk_level",
        "confidence", "low_confidence_flag",
        "prob_No DR", "prob_Mild NPDR", "prob_Moderate NPDR",
        "prob_Severe NPDR", "prob_Proliferative DR",
        "timestamp"
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {
                "image": r["image"],
                "predicted_stage": r["predicted_stage"],
                "predicted_label": r["predicted_label"],
                "risk_level": r["risk_level"],
                "confidence": r["confidence"],
                "low_confidence_flag": r["low_confidence_flag"],
                "timestamp": r["timestamp"],
            }
            for label, prob in r["probabilities"].items():
                row[f"prob_{label}"] = prob
            writer.writerow(row)
    print(f"  CSV report saved  → {path}")

# -------------------
# Main
# -------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(format_separator("═"))
print("  DR DIAGNOSIS TOOL — Batch Inference")
print(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(format_separator("═"))

# Load model + scaler
clf    = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Verify predict_proba is available
if not hasattr(clf, "predict_proba"):
    sys.exit(
        "The loaded classifier does not support predict_proba(). "
        "Re-train with a model that outputs probabilities (e.g. RandomForest, "
        "LogisticRegression, GradientBoosting) or set probability=True for SVC."
    )

print(f"  Model loaded  : {MODEL_PATH}")
print(f"  Scaler loaded : {SCALER_PATH} (expects {scaler.n_features_in_} features)")

# Load RETFound
weights = Path("models/checkpoints/retfound_cfp_vit_large_clean.pth")
loader  = RETFoundLoader(weights, device="cpu")
retfound_model = loader.load()
extractor = RETFoundExtractor(retfound_model)

# -------------------
# Extract features
# -------------------
image_files = sorted(os.listdir(TEST_IMAGES_DIR))
X_test = []

print(f"\n  Processing {len(image_files)} image(s) from {TEST_IMAGES_DIR}\n")

for img_file in image_files:
    img_path = os.path.join(TEST_IMAGES_DIR, img_file)
    img = Image.open(img_path).convert("RGB")
    features = extractor.extract(np.array(img))
    features = np.array(features).flatten()
    X_test.append(features)

X_test = np.array(X_test)
X_test_scaled = scaler.transform(X_test)

# -------------------
# Predict
# -------------------
y_pred  = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)   # shape: (n_images, 5)

# -------------------
# Per-image report
# -------------------
records = []
print("\n  PER-IMAGE REPORTS\n")

for img_file, pred, proba in zip(image_files, y_pred, y_proba):
    print_image_report(img_file, pred, proba, severity_map)
    records.append(build_record(img_file, pred, proba))

# -------------------
# Batch summary
# -------------------
print_summary(records)

# -------------------
# Export
# -------------------
print()
if EXPORT_JSON:
    export_json(records, OUTPUT_DIR)
if EXPORT_CSV:
    export_csv(records, OUTPUT_DIR)
