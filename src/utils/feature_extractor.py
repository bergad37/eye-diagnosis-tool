from __future__ import annotations

from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class RETFoundExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # RETFound input size
            transforms.ToTensor(),          # HWC -> CHW
            transforms.Normalize(
                # Keep consistent with other scripts (ImageNet normalization)
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Returns a 1D numpy feature vector from the backbone.
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        x = self.transform(image).unsqueeze(0)
        x = x.to(next(self.model.parameters()).device)

        with torch.no_grad():
            # Your RETFound model includes `.backbone`; use it as feature extractor
            features = self.model.backbone(x)

        # Handle different backbone outputs
        if len(features.shape) == 3:
            features = features[:, 0, :]  # CLS token
        elif len(features.shape) != 2:
            raise ValueError(f"Unexpected feature shape {features.shape}")

        return features.squeeze().cpu().numpy()
