import sys
from pathlib import Path
import torch

# 1️⃣ Add project root to Python path so src.* imports resolve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.models.retfound import RETFoundLoader

weights_path = Path("models/checkpoints/retfound_cfp_vit_large_clean.pth")

device = "mps" if torch.backends.mps.is_available() else "cpu"

loader = RETFoundLoader(weights_path, device=device)
model = loader.load()

print("Model loaded successfully!")
