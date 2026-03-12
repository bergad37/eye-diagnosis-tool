from timm import create_model
import torch
import torch.nn as nn

def vit_large_patch16(pretrained=True):
    # backbone only, no classifier
    return create_model("vit_large_patch16_224", num_classes=0, global_pool='avg', pretrained=pretrained)

class RETFoundOrdinal(nn.Module):
    def __init__(self, backbone=None, num_stages=5, hidden_dim=1024):
        """
        num_stages: total DR stages (0-4)
        hidden_dim: optional hidden layer before ordinal head
        """
        super().__init__()
        self.backbone = backbone or vit_large_patch16()

        # ordinal head: num_stages-1 thresholds
        self.ordinal_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_stages-1)  # 4 outputs for 5 stages
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.ordinal_head(features)  # [batch, 4]
        return logits

def build_retfound_model(pretrained=True):
    backbone = vit_large_patch16(pretrained=pretrained)
    model = RETFoundOrdinal(backbone=backbone)
    return model
