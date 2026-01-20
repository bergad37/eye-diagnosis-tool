import torch

class RETFoundWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def extract_features(self, images: torch.Tensor):
        with torch.no_grad():
            return self.model(images)
