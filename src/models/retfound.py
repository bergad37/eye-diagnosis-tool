import sys
from pathlib import Path

# Add project/src folder to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
from pathlib import Path

from .retfound_arch import build_retfound_model


class RETFoundLoader:
    def __init__(self, weights_path: Path, device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device
        self.model = None

    def load(self):
        if not self.weights_path.exists():
            raise FileNotFoundError(f"RETFound weights not found at {self.weights_path}")

        model = build_retfound_model()

        checkpoint = torch.load(self.weights_path, map_location=self.device)

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)

        model.to(self.device)
        model.eval()

        self.model = model
        return self.model

