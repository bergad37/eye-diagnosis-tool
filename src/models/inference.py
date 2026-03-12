import sys
from pathlib import Path

# Add project root to sys.path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from PIL import Image
from torchvision import transforms

from src.models.retfound import RETFoundLoader  # your loader

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use absolute path relative to project root
WEIGHTS_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "checkpoints" / "retfound_cfp_vit_large_clean.pth"
IMAGE_SIZE = 224  # ViT input size

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
def predict_stage(image_path: str):
    """
    image_path: path to a single fundus image
    returns: predicted DR stage (0-4)
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # add batch dim
    x = x.to(DEVICE)

    # Load model
    loader = RETFoundLoader(weights_path=WEIGHTS_PATH, device=DEVICE)
    model = loader.load()

    # Forward pass
    with torch.no_grad():
        logits = model(x)  # shape [1, 4] for ordinal
        probs = torch.sigmoid(logits)  # convert logits to probabilities
        stage = (probs > 0.5).sum(dim=1).item()  # final stage 0-4

    return stage

# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    test_image = "data/aptos/test_images/0a262e8b2a5a.png"  # replace with your image
    pred = predict_stage(test_image)
    print(f"Predicted DR Stage: {pred}")
