import torch
from torchvision import transforms
from PIL import Image

class RETFoundExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # RETFound input size
            transforms.ToTensor(),          # HWC -> CHW
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def extract(self, image):
     if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

        x = self.transform(image).unsqueeze(0)
        x = x.to(next(self.model.parameters()).device)

        with torch.no_grad():
         features = self.model.backbone(x)


    # Handle different backbone outputs
        if len(features.shape) == 3:
         features = features[:, 0, :]  # CLS token
        elif len(features.shape) == 2:
         features = features
        else:
          raise ValueError(f"Unexpected feature shape {features.shape}")

        return features.squeeze().cpu().numpy()
