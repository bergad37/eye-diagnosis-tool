from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
except ModuleNotFoundError:
    sys.exit(
        "joblib is not available. "
        "Activate your virtualenv or run `pip install -r requirements.txt`."
    )

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.retfound import RETFoundLoader
from src.utils.feature_extractor import RETFoundExtractor


@dataclass(frozen=True)
class TrainMetadata:
    created_at_utc: str
    dataset_csv: str
    images_dir: str
    weights_path: str
    image_size: int
    normalize_mean: Tuple[float, float, float]
    normalize_std: Tuple[float, float, float]
    model_type: str
    scaler_type: str
    classifier_type: str
    classifier_params: Dict[str, object]
    n_samples: int
    n_features: int
    label_counts: Dict[str, int]
    train_size: int
    val_size: int
    random_state: int


def _load_xy(
    csv_path: Path,
    images_dir: Path,
    extractor: RETFoundExtractor,
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "id_code" not in df.columns or "diagnosis" not in df.columns:
        raise ValueError("CSV must contain columns: id_code, diagnosis")

    x_list = []
    y_list = []

    # Lazy import to keep the top-level imports minimal
    from PIL import Image

    for _, row in df.iterrows():
        image_id = str(row["id_code"])
        label = int(row["diagnosis"])

        # APTOS images are commonly .png
        img_path = images_dir / f"{image_id}.png"
        if not img_path.exists():
            # fall back to jpg/jpeg if needed
            jpg = images_dir / f"{image_id}.jpg"
            jpeg = images_dir / f"{image_id}.jpeg"
            if jpg.exists():
                img_path = jpg
            elif jpeg.exists():
                img_path = jpeg
            else:
                continue

        img = Image.open(img_path).convert("RGB")
        feats = extractor.extract(img)  # shape [D]
        x_list.append(feats.astype(np.float32, copy=False))
        y_list.append(label)

    if not x_list:
        raise ValueError(
            "No training samples found. Check your CSV paths and images_dir."
        )

    X = np.vstack(x_list)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DR classifier (RETFound features + scaler + sklearn classifier)."
    )
    parser.add_argument("--csv", type=str, default="data/aptos/train.csv")
    parser.add_argument("--images", type=str, default="data/aptos/train_images")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/checkpoints/retfound_cfp_vit_large_clean.pth",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--max-iter", type=int, default=5000)
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    images_dir = PROJECT_ROOT / args.images
    weights_path = PROJECT_ROOT / args.weights
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load RETFound feature extractor model
    loader = RETFoundLoader(weights_path=weights_path, device=args.device)
    retfound_model = loader.load()
    extractor = RETFoundExtractor(retfound_model)

    # 2) Build dataset (X,y)
    X, y = _load_xy(csv_path=csv_path, images_dir=images_dir, extractor=extractor)

    # 3) Split
    X_train, x_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # 4) Scale features
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(X_train)
    x_val_s = scaler.transform(x_val)

    # 5) Train classifier (multinomial logistic regression)
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

    clf = LogisticRegression(
        max_iter=args.max_iter,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight=class_weight_dict,
    )
    clf.fit(x_train_s, y_train)

    # 6) Evaluate
    preds = clf.predict(x_val_s)
    report = classification_report(y_val, preds, digits=4)
    print(report)

    # 7) Save artifacts
    classifier_path = out_dir / "aptos_classifier.pkl"
    scaler_path = out_dir / "scaler.pkl"
    metadata_path = out_dir / "training_metadata.json"

    joblib.dump(clf, classifier_path)
    joblib.dump(scaler, scaler_path)

    counts = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}

    metadata = TrainMetadata(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_csv=str(Path(args.csv)),
        images_dir=str(Path(args.images)),
        weights_path=str(Path(args.weights)),
        image_size=224,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        model_type="RETFound (ViT-Large backbone features)",
        scaler_type="sklearn.preprocessing.StandardScaler",
        classifier_type="sklearn.linear_model.LogisticRegression(multi_class=multinomial)",
        classifier_params={
            "max_iter": args.max_iter,
            "solver": "lbfgs",
            "class_weight": class_weight_dict,
        },
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        label_counts=counts,
        train_size=int(X_train.shape[0]),
        val_size=int(x_val.shape[0]),
        random_state=int(args.random_state),
    )

    metadata_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")

    print(f"✅ Saved classifier: {classifier_path}")
    print(f"✅ Saved scaler:     {scaler_path}")
    print(f"✅ Saved metadata:   {metadata_path}")


if __name__ == "__main__":
    main()

