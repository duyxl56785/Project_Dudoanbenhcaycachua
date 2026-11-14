"""Systematic NV5 search for traditional HOG/SVM models.

Only Train and Validation are read during the search.  This script never opens
the fixed Test split; a separately invoked evaluator is responsible for that.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


LABELS = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot"]
FEATURES = {
    "gray64": {"size": 64, "grayscale": True, "orientations": 9, "ppc": 8},
    "rgb64": {"size": 64, "grayscale": False, "orientations": 9, "ppc": 8},
    "gray96": {"size": 96, "grayscale": True, "orientations": 9, "ppc": 8},
    "rgb64_orient12": {"size": 64, "grayscale": False, "orientations": 12, "ppc": 8},
    "rgb64_hsv": {"size": 64, "grayscale": False, "orientations": 9, "ppc": 8, "hsv": True},
}


def read_split(root: Path, data_root: Path, split: str) -> list[dict[str, str]]:
    with (root / "data" / "splits" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty split: {split}")
    for row in rows:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts or row["label"] not in {"0", "1", "2", "3"}:
            raise ValueError(f"Invalid row in {split}: {row}")
        if not (data_root / rel).is_file():
            raise FileNotFoundError(data_root / rel)
    return rows


def hsv_features(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    hsv = rgb2hsv(rgb).astype(np.float32)
    parts: list[np.ndarray] = []
    for channel in range(3):
        parts.append(np.histogram(hsv[:, :, channel], bins=bins, range=(0.0, 1.0), density=True)[0].astype(np.float32))
    size = rgb.shape[0]
    for y in range(4):
        for x in range(4):
            patch = hsv[y * size // 4 : (y + 1) * size // 4, x * size // 4 : (x + 1) * size // 4]
            parts.extend((patch.mean(axis=(0, 1)).astype(np.float32), patch.std(axis=(0, 1)).astype(np.float32)))
    return np.concatenate(parts)


def extract(rows: list[dict[str, str]], data_root: Path, feature_name: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = FEATURES[feature_name]
    vectors: list[np.ndarray] = []
    labels = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        with Image.open(data_root / row["path"]) as image:
            image_rgb = np.asarray(image.convert("RGB").resize((cfg["size"], cfg["size"]), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        pixels = image_rgb if not cfg["grayscale"] else np.asarray(Image.fromarray((image_rgb * 255).astype(np.uint8)).convert("L"), dtype=np.float32) / 255.0
        descriptor = hog(pixels, orientations=cfg["orientations"], pixels_per_cell=(cfg["ppc"], cfg["ppc"]), cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True, channel_axis=None if cfg["grayscale"] else -1).astype(np.float32)
        if cfg.get("hsv"):
            descriptor = np.concatenate([descriptor, hsv_features(image_rgb)])
        vectors.append(descriptor)
        labels[index] = int(row["label"])
    return np.vstack(vectors), labels


def score(y_true: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    return float(accuracy_score(y_true, prediction)), float(f1_score(y_true, prediction, labels=[0, 1, 2, 3], average="macro", zero_division=0))


def trial(feature_name: str, model_type: str, c_value: float, gamma: str | float | None, pca_components: int | None, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *, already_scaled: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    if already_scaled:
        train_scaled, val_scaled = x_train, x_val
    else:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(x_train)
        val_scaled = scaler.transform(x_val)
    pca = None
    if pca_components is not None:
        pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=42)
        train_scaled = pca.fit_transform(train_scaled)
        val_scaled = pca.transform(val_scaled)
    if model_type.endswith("linear"):
        model = LinearSVC(C=c_value, dual=False, max_iter=2000, tol=1e-2, random_state=42)
    else:
        model = SVC(C=c_value, kernel="rbf", gamma=(gamma or "scale"), cache_size=2048)
    model.fit(train_scaled, y_train)
    prediction = model.predict(val_scaled)
    accuracy, macro_f1 = score(y_val, prediction)
    return {
        "feature": feature_name,
        "model": model_type,
        "C": c_value,
        "gamma": gamma or "scale",
        "pca_components": pca_components or "none",
        "feature_dim": int(x_train.shape[1]),
        "val_accuracy": accuracy,
        "val_macro_f1": macro_f1,
        "train_seconds": time.perf_counter() - started,
        "scaler_fit": "Train only",
        "pca_fit": "Train only" if pca_components is not None else "none",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/traditional_svm_search"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    train_rows = read_split(root, data_root, "train")
    val_rows = read_split(root, data_root, "val")
    if len(train_rows) != 4115 or len(val_rows) != 1031:
        raise ValueError("Unexpected fixed Train/Validation counts")
    results: list[dict[str, object]] = []
    feature_best: list[dict[str, object]] = []

    # Stage 1: broad LinearSVC screening (20 configurations).
    for feature_name in FEATURES:
        print(f"extract {feature_name}", flush=True)
        x_train, y_train = extract(train_rows, data_root, feature_name)
        x_val, y_val = extract(val_rows, data_root, feature_name)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        local: list[dict[str, object]] = []
        for c_value in (0.01, 0.1, 1.0, 10.0):
            row = trial(feature_name, "hog_linear", c_value, "scale", None, x_train, y_train, x_val, y_val, already_scaled=True)
            local.append(row)
            results.append(row)
            print(f"  {feature_name} Linear C={c_value:g} val_f1={row['val_macro_f1']:.6f}", flush=True)
        feature_best.append(max(local, key=lambda r: (float(r["val_macro_f1"]), float(r["val_accuracy"]))))
        del x_train, x_val, y_train, y_val, scaler
        gc.collect()

    # Keep the two best validation feature families.  Together with the five
    # Stage-1 families this reproduces the 38 recorded Validation trials.
    top_features = [str(row["feature"]) for row in sorted(feature_best, key=lambda r: (float(r["val_macro_f1"]), float(r["val_accuracy"])), reverse=True)[:2]]
    print(f"top_features={top_features}", flush=True)

    # Stage 2: nonlinear RBF around the two best HOG feature families (4).
    for feature_name in top_features:
        x_train, y_train = extract(train_rows, data_root, feature_name)
        x_val, y_val = extract(val_rows, data_root, feature_name)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        for c_value in (1.0, 10.0):
            row = trial(feature_name, "hog_rbf", c_value, "scale", None, x_train, y_train, x_val, y_val, already_scaled=True)
            results.append(row)
            print(f"  {feature_name} RBF C={c_value:g} val_f1={row['val_macro_f1']:.6f}", flush=True)

        # Stage 3: PCA + LinearSVC (6 configurations per selected family; 12 total).
        for components in (64, 128, 256):
            for c_value in (0.1, 1.0):
                row = trial(feature_name, "hog_pca_linear", c_value, "scale", components, x_train, y_train, x_val, y_val, already_scaled=True)
                results.append(row)
                print(f"  {feature_name} PCA{components} Linear C={c_value:g} val_f1={row['val_macro_f1']:.6f}", flush=True)
        del x_train, x_val, y_train, y_val, scaler
        gc.collect()

    current_best = max(results, key=lambda r: (float(r["val_macro_f1"]), float(r["val_accuracy"]), -float(r["train_seconds"])))
    # Stage 4: PCA + RBF around the best observed feature family (2).
    best_feature = str(current_best["feature"])
    x_train, y_train = extract(train_rows, data_root, best_feature)
    x_val, y_val = extract(val_rows, data_root, best_feature)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    for c_value in (1.0, 10.0):
        row = trial(best_feature, "hog_pca_rbf", c_value, "scale", 128, x_train, y_train, x_val, y_val, already_scaled=True)
        results.append(row)
        print(f"  {best_feature} PCA128 RBF C={c_value:g} val_f1={row['val_macro_f1']:.6f}", flush=True)
    del x_train, x_val, y_train, y_val, scaler
    gc.collect()

    results_path = out / "validation_search.csv"
    with results_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    best = max(results, key=lambda r: (float(r["val_macro_f1"]), float(r["val_accuracy"]), -float(r["train_seconds"])))
    (out / "best_config.json").write_text(json.dumps({"selection": "highest Validation Macro-F1", "trials": len(results), "best": best}, indent=2) + "\n", encoding="utf-8")
    print(f"BEST {json.dumps(best)}", flush=True)
    print(f"TRIALS {len(results)}", flush=True)


if __name__ == "__main__":
    main()
