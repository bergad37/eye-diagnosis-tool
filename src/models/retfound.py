import sys
from pathlib import Path
import torch
import os

from huggingface_hub import hf_hub_download
from .retfound_arch import build_retfound_model
from typing import Optional

# Add project/src folder to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))


class RETFoundLoader:
    def __init__(
        self,
        device: str = "cpu",
       weights_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        self.device = device
        self.weights_path = weights_path
        self.repo_id = repo_id
        self.filename = filename
        self.model = None

    def load(self):
        print("🚀 Loading RETFound model...", flush=True)

        # ✅ Decide source
        if self.weights_path:
            print(f"📁 Loading RETFound weights from local file: {self.weights_path}", flush=True)
            weights_path = self.weights_path

        elif self.repo_id and self.filename:
            print(f"⬇️ Downloading RETFound weights from Hugging Face: {self.repo_id}/{self.filename}", flush=True)
            weights_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
            )
            print(f"✅ Download complete: {weights_path}", flush=True)

        else:
            raise ValueError(
                "Provide either weights_path OR (repo_id + filename)"
            )

        # ✅ Build model
        # Avoid extra timm pretrained download; we load our own checkpoint.
        model = build_retfound_model(pretrained=False)

        print("📦 Loading checkpoint with torch.load...", flush=True)
        checkpoint = torch.load(weights_path, map_location=self.device)

        state_dict = checkpoint.get("model", checkpoint)

        model.load_state_dict(state_dict, strict=False)

        model.to(self.device)
        model.eval()

        self.model = model

        print("✅ Model loaded successfully", flush=True)
        return model