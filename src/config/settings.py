from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
CHECKPOINTS = MODEL_DIR / "checkpoints"
OUTPUTS = MODEL_DIR / "outputs"

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 2
DEVICE = "cuda"  # fallback to cpu later
