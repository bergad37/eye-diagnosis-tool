import sys
from pathlib import Path
import torch
import os

from huggingface_hub import hf_hub_download
from .retfound_arch import build_retfound_model

# Add project/src folder to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))


class RETFoundLoader:
    def __init__(self, repo_id: str, filename: str, device: str = "cpu"):
        self.repo_id = repo_id
        self.filename = filename
        self.device = device
        self.model = None

    def load(self):
        print("⬇️ Downloading / loading RETFound weights from HuggingFace...")

        weights_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename
        )

        model = build_retfound_model()

        checkpoint = torch.load(weights_path, map_location=self.device)

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)

        model.to(self.device)
        model.eval()

        self.model = model
        print("✅ Model loaded successfully")

        return self.model