"""Evaluate the frozen NV5 HSV-enhanced HOG/RBF-SVM on the fixed Test split.

Selection was completed on Train/Validation only (see the validation artifact).
This script fits on Train, opens Test only for the final prediction pass, and
emits auditable predictions and metrics without changing NV4's baseline files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_splits(root: Path, data_root: Path) -> dict[str, list[dict[str, str]]]:
    splits: dict[str, list[dict[str, str]]] = {}
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        with (root / "data" / "splits" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty {split} split")
        for row in rows:
            path = Path(row["path"])
            if path.is_absolute() or ".." in path.parts or row["path"] in seen:
                raise ValueError(f"Unsafe or duplicated path: {row['path']}")
            if row["label"] not in {"0", "1", "2", "3"}:
                raise ValueError(f"Invalid class index: {row['label']}")
            if not (data_root / path).is_file():
                raise FileNotFoundError(data_root / path)
            seen.add(row["path"])
        splits[split] = rows
    return splits


def load_mapping(root: Path) -> tuple[str, ...]:
    mapping = json.loads((root / "configs" / "class_mapping.json").read_text(encoding="utf-8"))
    names = tuple(mapping[str(i)] for i in range(4))
    if len(set(names)) != 4:
        raise ValueError("Class names must be unique")
    return names


def hsv_color_features(path: Path, size: int = 64, bins: int = 16) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    hsv = rgb2hsv(rgb).astype(np.float32)
    parts: list[np.ndarray] = []
    for channel in range(3):
        parts.append(np.histogram(hsv[:, :, channel], bins=bins, range=(0.0, 1.0), density=True)[0].astype(np.float32))
    for y in range(4):
        for x in range(4):
            patch = hsv[y * size // 4 : (y + 1) * size // 4, x * size // 4 : (x + 1) * size // 4]
            parts.extend((patch.mean(axis=(0, 1)).astype(np.float32), patch.std(axis=(0, 1)).astype(np.float32)))
    return np.concatenate(parts)


def feature_matrix(rows: list[dict[str, str]], data_root: Path, config) -> tuple[np.ndarray, np.ndarray]:
    # Import NV4's deterministic HOG implementation read-only.
    vectors: list[np.ndarray] = []
    labels = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        path = data_root / row["path"]
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        from skimage.feature import hog
        descriptor = hog(rgb, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True, channel_axis=-1).astype(np.float32)
        vectors.append(np.concatenate([descriptor, hsv_color_features(path)]))
        labels[index] = int(row["label"])
    return np.vstack(vectors), labels


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/hog_svm_nv5_experiment"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(root, args.data_root.resolve())
    names = load_mapping(root)
    # Train only; validation was used before this script to freeze C=10.
    x_train, y_train = feature_matrix(splits["train"], args.data_root.resolve(), config)
    x_test, y_test = feature_matrix(splits["test"], args.data_root.resolve(), config)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    classifier = SVC(C=10.0, kernel="rbf", gamma="scale", cache_size=2048)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    labels = [0, 1, 2, 3]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    metrics = {
        "model": "HOG + HSV colour features + RBF SVM",
        "selection": "Validation macro-F1; Test opened only for final evaluation",
        "feature": {"image_size": 64, "hog_grayscale": False, "orientations": 9, "pixels_per_cell": 8, "cells_per_block": 2, "block_norm": "L2-Hys", "hsv_bins": 16, "hsv_spatial_grid": "4x4", "dimension": int(x_train.shape[1])},
        "C": 10.0,
        "gamma": "scale",
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "test_correct": int(np.sum(y_test == predictions)),
        "test_errors": int(np.sum(y_test != predictions)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(precision_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {names[i]: float(v) for i, v in enumerate(f1_score(y_test, predictions, labels=labels, average=None, zero_division=0))},
        "confusion_matrix": matrix.tolist(),
        "class_names": list(names),
    }
    joblib.dump({"scaler": scaler, "classifier": classifier, "class_names": names, "config": metrics["feature"], "C": 10.0}, out / "hog_hsv_rbf_svm.joblib", compress=3)
    prediction_path = out / "test_predictions.csv"
    fields = ["image_id", "path", "true_label_index", "true_label", "predicted_label_index", "predicted_label", "model", "status"]
    with prediction_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row, true_label, pred_label in zip(splits["test"], y_test, predictions):
            writer.writerow({"image_id": Path(row["path"]).stem, "path": row["path"], "true_label_index": int(true_label), "true_label": names[int(true_label)], "predicted_label_index": int(pred_label), "predicted_label": names[int(pred_label)], "model": "hog_hsv_rbf_svm", "status": "ok"})
    metrics["prediction_csv_sha256"] = sha256(prediction_path)
    metrics["checkpoint_sha256"] = sha256(out / "hog_hsv_rbf_svm.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
