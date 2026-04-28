from timm import create_model
import torch.nn as nn


def vit_large_patch16(pretrained: bool = True):
    # Backbone only, no classifier head.
    return create_model(
        "vit_large_patch16_224",
        num_classes=0,
        global_pool="avg",
        pretrained=pretrained,
    )


class RETFoundOrdinal(nn.Module):
    def __init__(self, backbone=None, num_stages: int = 5, hidden_dim: int = 1024):
        super().__init__()
        self.backbone = backbone or vit_large_patch16(pretrained=False)
        self.ordinal_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_stages - 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.ordinal_head(features)
        return logits


def build_retfound_model(pretrained: bool = True):
    backbone = vit_large_patch16(pretrained=pretrained)
    return RETFoundOrdinal(backbone=backbone)

