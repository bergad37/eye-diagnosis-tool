import sys
from pathlib import Path

# Add project root to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from src.models.retfound import RETFoundLoader


DEVICE = "cpu"  # safer on Mac, change later if needed

DATA_DIR = Path("data/aptos")
IMAGES_DIR = DATA_DIR / "train_images"
CSV_PATH = DATA_DIR / "train.csv"

FEATURES_OUT = Path("data/aptos_features.npz")


# ---------------------------
# Image preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


# ---------------------------
# Load RETFound
# ---------------------------
loader = RETFoundLoader(
    Path("models/checkpoints/retfound_cfp_vit_large_clean.pth"),
    device=DEVICE
)
model = loader.load()


# ---------------------------
# Load labels
# ---------------------------
df = pd.read_csv(CSV_PATH)

features = []
labels = []


# ---------------------------
# Extract features
# ---------------------------
with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = IMAGES_DIR / f"{row['id_code']}.png"

        if not img_path.exists():
            continue

        img_tensor = load_image(img_path).to(DEVICE)

        feat = model(img_tensor)
        feat = feat.squeeze().cpu().numpy()

        features.append(feat)
        labels.append(row["diagnosis"])


features = np.array(features)
labels = np.array(labels)

np.savez(FEATURES_OUT, X=features, y=labels)

print("✅ Features saved:", FEATURES_OUT)
print("Feature shape:", features.shape)
