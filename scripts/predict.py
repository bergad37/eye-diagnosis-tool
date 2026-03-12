import os
import sys
import numpy as np

# joblib may not be installed if the wrong Python interpreter is used; give
# a clear message so users know to activate their virtualenv or install
# requirements.
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

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.feature_extractor import RETFoundExtractor
from src.models.retfound import RETFoundLoader

# -------------------
# Config
# -------------------
MODEL_PATH = "models/aptos_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"
TEST_IMAGES_DIR = "data/aptos/test_images/"

# DR severity labels
severity_map = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# -------------------
# Load trained model and scaler
# -------------------
clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Scaler expects features:", scaler.n_features_in_)
# -------------------
# Load RETFound model (feature extractor)
# -------------------
weights = Path("models/checkpoints/retfound_cfp_vit_large_clean.pth")
loader = RETFoundLoader(weights, device="cpu")
retfound_model = loader.load()

extractor = RETFoundExtractor(retfound_model)

# -------------------
# Load images and extract features
# -------------------
image_files = sorted(os.listdir(TEST_IMAGES_DIR))
X_test = []

for img_file in image_files:
    img_path = os.path.join(TEST_IMAGES_DIR, img_file)
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)

    features = extractor.extract(img_array)

print("Raw feature shape:", np.array(features).shape)

features = np.array(features).flatten()

print("Flattened feature shape:", features.shape)

X_test.append(features)

X_test = np.array(X_test)

# -------------------
# Scale features
# -------------------
X_test_scaled = scaler.transform(X_test)

# -------------------
# Predict DR stage
# -------------------
y_pred = clf.predict(X_test_scaled)

# -------------------
# Print predictions per image
# -------------------
predictions = []
print("\nPredictions:")
for img_file, pred in zip(image_files, y_pred):
    print(f"{img_file}: Predicted DR stage = {pred} ({severity_map[pred]})")
    predictions.append(pred)

# -------------------
# Print counts of each predicted class
# -------------------
pred_counts = Counter(predictions)
print("\nPrediction counts:")
for cls, count in pred_counts.items():
    print(f"{severity_map[cls]}: {count}")
