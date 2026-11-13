"""Run a separate Validation-selected HOG + LinearSVC development round."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
import skimage
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from scripts.train_hog_svm import sha256, write_json
from src.hog_svm import HOGConfig, config_dict, feature_matrix, load_mapping, read_splits


CONFIGS = (
    HOGConfig(image_size=64, grayscale=False, orientations=9, pixels_per_cell=8),
    HOGConfig(image_size=96, orientations=12, pixels_per_cell=16),
    HOGConfig(image_size=96, grayscale=False, orientations=9, pixels_per_cell=16),
    HOGConfig(image_size=128, orientations=9, pixels_per_cell=16),
)
C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0)
WEIGHTS = (None, "balanced")
LABELS = (0, 1, 2, 3)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision), "macro_recall": float(recall),
            "macro_f1": float(f1)}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hog_svm_v2"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    output = args.artifact_dir.resolve()
    if output.exists() and any(path.name != "preliminary_gray96_high_dim" for path in output.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing V2 evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)

    names = load_mapping(project)
    splits = read_splits(project, data_root)
    counts = {name: len(rows) for name, rows in splits.items()}
    if counts != {"train": 4115, "val": 1031, "test": 1356}:
        raise ValueError(f"Unexpected split counts: {counts}")
    split_hashes = {name: sha256(project / "data/splits" / f"{name}.csv") for name in splits}
    v1 = json.loads((project / "results/hog_svm/model_metadata.json").read_text(encoding="utf-8"))
    if counts != v1["split_counts"] or split_hashes != v1["split_csv_sha256"] or list(names) != v1["class_names"]:
        raise ValueError("V2 split or class mapping differs from verified V1")

    log = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
           "split_counts": counts, "split_csv_sha256": split_hashes,
           "class_names": names, "seed": 42, "scaler_fit": "Train only",
           "selection_rule": "converged trials only; highest Validation macro-F1; then faster Validation prediction; then earlier trial",
           "test_usage": "single final evaluation after selection; V1 Test was already visible before V2",
           "versions": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_image": skimage.__version__, "scikit_learn": sklearn.__version__,
                        "joblib": joblib.__version__},
           "feature_extraction": [], "experiments": []}
    write_json(output / "train_log.json", log)
    best = None
    for config in CONFIGS:
        matrices = {}
        labels = {}
        for split in ("train", "val"):
            started = time.perf_counter()
            matrices[split], labels[split] = feature_matrix(splits[split], data_root, config)
            if not np.isfinite(matrices[split]).all():
                raise ValueError(f"Non-finite {split} HOG features")
            extraction = {"config": config_dict(config), "split": split,
                          "rows": len(labels[split]), "dimension": config.dimension(),
                          "matrix_bytes": matrices[split].nbytes,
                          "seconds": time.perf_counter() - started}
            log["feature_extraction"].append(extraction)
            write_json(output / "train_log.json", log)
            print("HOG", config, split, extraction["seconds"], "seconds", flush=True)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(matrices["train"])
        x_val = scaler.transform(matrices["val"])
        if int(scaler.n_samples_seen_) != counts["train"]:
            raise RuntimeError("Scaler was not fitted exclusively on Train")
        del matrices
        for c_value in C_VALUES:
            for weight in WEIGHTS:
                classifier = LinearSVC(C=c_value, class_weight=weight, dual=False,
                                       max_iter=150, tol=1e-2, random_state=42)
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    classifier.fit(x_train, labels["train"])
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                predicted = classifier.predict(x_val)
                predict_ms = (time.perf_counter() - started) * 1000 / counts["val"]
                scores = metrics(labels["val"], predicted)
                trial = {"trial": len(log["experiments"]) + 1, **config_dict(config),
                         "hog_dimension": config.dimension(), "C": c_value,
                         "class_weight": weight or "none", "max_iter": 150, "tol": 1e-2,
                         "train_seconds": fit_seconds, "val_predict_ms_per_image": predict_ms,
                         "iterations": int(np.max(classifier.n_iter_)),
                         "convergence_warnings": len([w for w in caught if issubclass(w.category, ConvergenceWarning)]),
                         "val_accuracy": scores["accuracy"], "val_macro_precision": scores["macro_precision"],
                         "val_macro_recall": scores["macro_recall"], "val_macro_f1": scores["macro_f1"]}
                log["experiments"].append(trial)
                trial["eligible_for_selection"] = trial["convergence_warnings"] == 0
                write_csv(output / "validation_experiments.csv", log["experiments"])
                write_json(output / "train_log.json", log)
                print("trial", trial["trial"], "val_f1", scores["macro_f1"],
                      "fit_s", round(fit_seconds, 2), "warnings", trial["convergence_warnings"], flush=True)
                rank = (scores["macro_f1"], -predict_ms, -trial["trial"])
                if trial["eligible_for_selection"] and (best is None or rank > best[0]):
                    best = (rank, config, scaler, classifier, trial)

    assert best is not None
    _, config, scaler, classifier, selected = best
    checkpoint = output / "hog_linear_svm_v2.joblib"
    bundle = {"hog_config": config_dict(config), "scaler": scaler, "classifier": classifier,
              "class_names": names, "C": selected["C"], "class_weight": selected["class_weight"],
              "split_csv_sha256": split_hashes}
    joblib.dump(bundle, checkpoint, compress=3)
    loaded = joblib.load(checkpoint)
    sample = feature_matrix(splits["val"][:1], data_root, config)[0]
    if not np.array_equal(classifier.predict(scaler.transform(sample)),
                          loaded["classifier"].predict(loaded["scaler"].transform(sample))):
        raise RuntimeError("Checkpoint reload changes predictions")

    # Test data is transformed and evaluated only after the Validation winner is frozen.
    started = time.perf_counter()
    x_test, y_test = feature_matrix(splits["test"], data_root, config)
    extraction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scaled_test = loaded["scaler"].transform(x_test)
    predictions = loaded["classifier"].predict(scaled_test)
    decision_scores = loaded["classifier"].decision_function(scaled_test)
    inference_seconds = time.perf_counter() - started
    prediction_rows = []
    for row, truth, predicted, scores in zip(splits["test"], y_test, predictions, decision_scores):
        prediction_rows.append({"image_id": Path(row["path"]).stem, "path": row["path"],
                                "true_label_index": int(truth), "true_label": names[int(truth)],
                                "predicted_label_index": int(predicted), "predicted_label": names[int(predicted)],
                                "model": checkpoint.name, "status": "ok", "error": "",
                                **{f"decision_score_{i}": float(score) for i, score in enumerate(scores)}})
    write_csv(output / "test_predictions.csv", prediction_rows)
    test_metrics = metrics(y_test, predictions)
    test_metrics.update({"n_samples": len(y_test), "correct": int(np.sum(y_test == predictions)),
                         "wrong": int(np.sum(y_test != predictions)),
                         "confusion_matrix": confusion_matrix(y_test, predictions, labels=LABELS).tolist(),
                         "test_hog_seconds": extraction_seconds, "test_predict_seconds": inference_seconds,
                         "test_predict_ms_per_image": inference_seconds * 1000 / len(y_test),
                         "test_total_ms_per_image": (extraction_seconds + inference_seconds) * 1000 / len(y_test),
                         "prediction_csv_sha256": sha256(output / "test_predictions.csv")})
    write_json(output / "test_metrics.json", test_metrics)
    metadata = {"model": "HOG + LinearSVC V2", "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
                "hog_config": config_dict(config), "hog_dimension": config.dimension(),
                "C": selected["C"], "class_weight": selected["class_weight"],
                "class_names": names, "split_counts": counts, "split_csv_sha256": split_hashes,
                "validation": selected, "selection_rule": log["selection_rule"],
                "scaler_fit": "Train only", "final_fit": "Train only; no Train+Val refit",
                "checkpoint_reload_prediction_equal": True, "versions": log["versions"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "test_prediction_rows": len(prediction_rows), "test_prediction_errors": 0,
                "test_predictions_sha256": test_metrics["prediction_csv_sha256"]}
    write_json(output / "model_metadata.json", metadata)
    log["selected_trial"] = selected["trial"]
    log["checkpoint_sha256"] = metadata["checkpoint_sha256"]
    log["completed_at_utc"] = metadata["completed_at_utc"]
    write_json(output / "train_log.json", log)
    print("Selected", selected["trial"], "Val macro-F1", selected["val_macro_f1"],
          "Test accuracy", test_metrics["accuracy"], "Test macro-F1", test_metrics["macro_f1"], flush=True)


if __name__ == "__main__":
    main()
