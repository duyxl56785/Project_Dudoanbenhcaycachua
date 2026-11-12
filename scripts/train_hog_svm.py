"""Train the HOG + LinearSVC baseline using NV1's fixed Train/Val/Test CSVs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
import skimage
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.hog_svm import HOGConfig, config_dict, feature_matrix, image_hog, load_mapping, read_splits


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing raw/color/... images")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hog_svm"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact_dir / "hog_linear_svm.joblib"
    if checkpoint.exists():
        raise SystemExit(f"Refusing to overwrite existing experiment: {checkpoint}")

    started_at = datetime.now(timezone.utc).isoformat()
    names = load_mapping(project_root)
    splits = read_splits(project_root, data_root)
    expected = {"train": 4115, "val": 1031, "test": 1356}
    counts = {split: len(rows) for split, rows in splits.items()}
    if counts != expected:
        raise ValueError(f"Unexpected split counts: {counts}")
    hashes = {split: sha256(project_root / "data/splits" / f"{split}.csv") for split in splits}
    configs = [HOGConfig(pixels_per_cell=8), HOGConfig(pixels_per_cell=16)]
    c_values = (0.01, 0.1, 1.0)
    experiments = []
    best = None
    log = {"started_at_utc": started_at, "split_counts": counts, "split_csv_sha256": hashes,
           "class_names": names, "selection_rule": "highest Val macro-F1; then lower Val prediction time; then earlier trial",
           "scaler_fit": "Train only", "test_usage": "Predictions only, after model selection",
           "versions": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_image": skimage.__version__, "scikit_learn": sklearn.__version__,
                        "joblib": joblib.__version__}, "feature_extraction": [], "experiments": experiments}
    write_json(artifact_dir / "train_log.json", log)

    for config_index, config in enumerate(configs):
        matrices = {}
        labels = {}
        for split in ("train", "val"):
            start = time.perf_counter()
            matrices[split], labels[split] = feature_matrix(splits[split], data_root, config)
            duration = time.perf_counter() - start
            log["feature_extraction"].append({"hog_config": config_dict(config), "split": split,
                                               "rows": len(labels[split]), "vector_size": config.dimension(),
                                               "seconds": duration})
            print(f"HOG config={config_index} {split}: {len(labels[split])} images, dim={config.dimension()}, {duration:.1f}s", flush=True)
            write_json(artifact_dir / "train_log.json", log)

        for c_value in c_values:
            start = time.perf_counter()
            scaler = StandardScaler()
            x_train = scaler.fit_transform(matrices["train"])
            x_val = scaler.transform(matrices["val"])
            classifier = LinearSVC(C=c_value, dual=False, max_iter=3000, random_state=42)
            classifier.fit(x_train, labels["train"])
            train_seconds = time.perf_counter() - start
            predict_start = time.perf_counter()
            predictions = classifier.predict(x_val)
            predict_ms = (time.perf_counter() - predict_start) * 1000 / len(predictions)
            accuracy = float(accuracy_score(labels["val"], predictions))
            macro_f1 = float(f1_score(labels["val"], predictions, average="macro", labels=[0, 1, 2, 3]))
            row = {"trial": len(experiments) + 1, "image_size": config.image_size,
                   "grayscale": config.grayscale, "orientations": config.orientations,
                   "pixels_per_cell": config.pixels_per_cell, "cells_per_block": config.cells_per_block,
                   "block_norm": config.block_norm, "hog_dimension": config.dimension(), "C": c_value,
                   "val_accuracy": accuracy, "val_macro_f1": macro_f1,
                   "train_seconds": train_seconds, "val_predict_ms_per_image": predict_ms,
                   "iterations": int(np.max(classifier.n_iter_))}
            experiments.append(row)
            with (artifact_dir / "validation_experiments.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(experiments)
            write_json(artifact_dir / "train_log.json", log)
            print(f"trial={row['trial']} C={c_value} Val accuracy={accuracy:.6f} macro-F1={macro_f1:.6f} train={train_seconds:.1f}s", flush=True)
            rank = (macro_f1, -predict_ms, -row["trial"])
            if best is None or rank > best[0]:
                best = (rank, config, c_value, scaler, classifier, row)

    assert best is not None
    _, config, c_value, scaler, classifier, selected = best
    # The selected estimator and scaler were fit on Train only. Val remains an independent selection set.
    bundle = {"hog_config": config_dict(config), "scaler": scaler, "classifier": classifier,
              "class_names": names, "C": c_value, "split_csv_sha256": hashes}
    joblib.dump(bundle, checkpoint, compress=3)
    checkpoint_hash = sha256(checkpoint)
    loaded = joblib.load(checkpoint)
    sample = image_hog(data_root / splits["val"][0]["path"], config).reshape(1, -1)
    before = classifier.predict(scaler.transform(sample))
    after = loaded["classifier"].predict(loaded["scaler"].transform(sample))
    if not np.array_equal(before, after):
        raise RuntimeError("Predictions differ after checkpoint reload")

    metadata = {"model": "HOG + LinearSVC", "checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_hash,
                "checkpoint_bytes": checkpoint.stat().st_size, "hog_config": config_dict(config),
                "hog_dimension": config.dimension(), "C": c_value, "class_names": names,
                "split_counts": counts, "split_csv_sha256": hashes, "validation": selected,
                "selection_rule": log["selection_rule"], "scaler_fit": "Train only",
                "final_fit": "Train only; no Train+Val refit", "versions": log["versions"],
                "checkpoint_reload_prediction_equal": True, "started_at_utc": started_at,
                "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(artifact_dir / "model_metadata.json", metadata)
    log["selected_trial"] = selected["trial"]
    log["checkpoint_sha256"] = checkpoint_hash
    write_json(artifact_dir / "train_log.json", log)

    # Test is opened only now. No Test label or score participates in configuration selection.
    test_rows = splits["test"]
    prediction_path = artifact_dir / "test_predictions.csv"
    fields = ["image_id", "path", "true_label_index", "true_label", "predicted_label_index",
              "predicted_label", "model", "hog_config", "C", "status", "error"] + [f"decision_score_{i}" for i in range(4)]
    errors = 0
    with prediction_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in test_rows:
            index = int(row["label"])
            record = {"image_id": Path(row["path"]).stem, "path": row["path"],
                      "true_label_index": index, "true_label": names[index], "model": checkpoint.name,
                      "hog_config": f"size={config.image_size};cell={config.pixels_per_cell};block={config.cells_per_block};orient={config.orientations};gray={config.grayscale};norm={config.block_norm}",
                      "C": c_value}
            try:
                vector = image_hog(data_root / row["path"], config).reshape(1, -1)
                scaled = loaded["scaler"].transform(vector)
                predicted = int(loaded["classifier"].predict(scaled)[0])
                scores = loaded["classifier"].decision_function(scaled)[0]
                record.update({"predicted_label_index": predicted, "predicted_label": names[predicted], "status": "ok", "error": ""})
                record.update({f"decision_score_{i}": float(score) for i, score in enumerate(scores)})
            except (OSError, ValueError) as exc:
                errors += 1
                record.update({"predicted_label_index": "", "predicted_label": "", "status": "error", "error": str(exc)})
            writer.writerow(record)
    metadata["test_prediction_rows"] = len(test_rows)
    metadata["test_prediction_errors"] = errors
    metadata["test_predictions_sha256"] = sha256(prediction_path)
    write_json(artifact_dir / "model_metadata.json", metadata)
    print(f"Selected trial={selected['trial']} Val macro-F1={selected['val_macro_f1']:.6f}; Test rows={len(test_rows)}, errors={errors}; SHA-256={checkpoint_hash}", flush=True)


if __name__ == "__main__":
    main()
