import torch
from pathlib import Path

src = Path("models/checkpoints/retfound_cfp_vit_large.pth")
dst = Path("models/checkpoints/retfound_cfp_vit_large_clean.pth")

# 🔓 Explicitly allow full loading (trusted checkpoint)
checkpoint = torch.load(
    src,
    map_location="cpu",
    weights_only=False
)

# RETFound checkpoints store weights under "model"
state_dict = checkpoint["model"]

# Save clean weights-only checkpoint
torch.save(state_dict, dst)

print("✅ Clean weights saved to:", dst)
