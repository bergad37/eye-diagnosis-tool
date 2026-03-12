# src/models/classifier.py
import torch.nn as nn

class DiagnosisClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)
