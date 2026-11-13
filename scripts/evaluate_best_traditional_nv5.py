"""Final Test evaluation for the frozen best NV5 traditional-model search row."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.search_traditional_models_nv5 import LABELS, FEATURES, extract, read_split


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, default=Path("results/traditional_svm_search"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    out = args.search_dir.resolve()
    best = json.loads((out / "best_config.json").read_text(encoding="utf-8"))["best"]
    if best["model"] != "hog_pca_rbf":
        raise ValueError(f"Unexpected best model: {best['model']}")
    train_rows = read_split(root, data_root, "train")
    test_rows = read_split(root, data_root, "test")
    x_train, y_train = extract(train_rows, data_root, best["feature"])
    x_test, y_test = extract(test_rows, data_root, best["feature"])
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    components = int(best["pca_components"])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=42)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    model = SVC(C=float(best["C"]), kernel="rbf", gamma="scale", cache_size=2048)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    labels = [0, 1, 2, 3]
    matrix = confusion_matrix(y_test, prediction, labels=labels)
    checkpoint = out / "best_traditional_svm.joblib"
    joblib.dump({"feature": FEATURES[best["feature"]], "scaler": scaler, "pca": pca, "classifier": model, "class_names": LABELS, "selection": best}, checkpoint, compress=3)
    csv_path = out / "test_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["image_id", "path", "true_label_index", "true_label", "predicted_label_index", "predicted_label", "model", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row, actual, predicted in zip(test_rows, y_test, prediction):
            writer.writerow({"image_id": Path(row["path"]).stem, "path": row["path"], "true_label_index": int(actual), "true_label": LABELS[int(actual)], "predicted_label_index": int(predicted), "predicted_label": LABELS[int(predicted)], "model": "hog_pca_rbf", "status": "ok"})
    metrics = {
        "selection_source": "results/traditional_svm_search/best_config.json",
        "selection_rule": (
            "highest Validation Macro-F1; this PCA result is exploratory because "
            "a separate non-PCA experiment had already opened Test"
        ),
        "selected_config": best,
        "split_counts": {"train": len(train_rows), "test": len(test_rows)},
        "test_correct": int(np.sum(y_test == prediction)),
        "test_errors": int(np.sum(y_test != prediction)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "macro_precision": float(precision_score(y_test, prediction, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, prediction, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, prediction, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {LABELS[i]: float(value) for i, value in enumerate(f1_score(y_test, prediction, labels=labels, average=None, zero_division=0))},
        "confusion_matrix": matrix.tolist(),
        "class_names": LABELS,
        "prediction_csv_sha256": sha256(csv_path),
        "checkpoint_sha256": sha256(checkpoint),
    }
    (out / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
