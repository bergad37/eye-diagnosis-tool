import sys
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torchvision import transforms

# ✅ Add src to path so we can import models reliably
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from models.retfound import RETFoundLoader
from models.retfound_arch import build_retfound_model  # ensure this works

# 1️⃣ Path to the cleaned RETFound checkpoint
weights_path = Path("models/checkpoints/retfound_cfp_vit_large_clean.pth")

# 2️⃣ Select device (MPS for Mac, else CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 3️⃣ Load the model
loader = RETFoundLoader(weights_path, device=device)
model = loader.load()
print("✅ RETFound model loaded successfully!")

# 4️⃣ Load and preprocess test eye image
img_path = Path("data/dummy_eye.jpg")  # <-- replace with your test image
if not img_path.exists():
    raise FileNotFoundError(f"Test image not found at {img_path}")

img = Image.open(img_path).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    ),
])

# Compose returns Tensor when pipeline ends with ToTensor(); cast for type checker
tensor = cast(torch.Tensor, preprocess(img))
input_tensor = tensor.unsqueeze(0).to(device)  # add batch dim and move to device

# 5️⃣ Run inference
with torch.no_grad():
    features = model(input_tensor)

# 6️⃣ Print output
print("✅ Feature vector shape:", features.shape)
print("Feature vector (first 10 values):", features[0, :10].cpu().numpy())
