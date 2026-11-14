"""Checks for the independently trained V2 artifact and fixed Test split."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from PIL import Image
from sklearn.metrics import confusion_matrix

from scripts.infer_hog_svm import predict_file
from scripts.train_hog_svm_v2 import CONFIGS, metrics
from src.hog_svm import HOGConfig, extract_hog, image_hog, load_mapping


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/hog_svm_v2"


def test_v2_config_dimensions_and_nonfinite_guard():
    expected = (1764, 1200, 900, 1764)
    for config, dimension in zip(CONFIGS, expected):
        image = Image.new("RGB", (150, 105), "green")
        vector = extract_hog(image, config)
        assert vector.shape == (dimension,)
        assert np.isfinite(vector).all()
    with pytest.raises(ValueError, match="Invalid HOG"):
        HOGConfig(image_size=96, pixels_per_cell=7).dimension()


def test_v2_metric_function_uses_macro_average():
    result = metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert result["accuracy"] == 0.75
    assert result["macro_recall"] == 0.375
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 4)


def test_published_v2_checkpoint_and_predictions_match_fixed_split(tmp_path):
    metadata = json.loads((V2 / "model_metadata.json").read_text(encoding="utf-8"))
    checkpoint = V2 / metadata["checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == metadata["checkpoint_sha256"]
    bundle = joblib.load(checkpoint)
    assert int(bundle["scaler"].n_samples_seen_) == 4115
    assert bundle["hog_config"] == metadata["hog_config"]
    assert tuple(bundle["class_names"]) == load_mapping(ROOT)
    assert metadata["checkpoint_reload_prediction_equal"] is True
    assert "highest Validation macro-F1" in metadata["selection_rule"]

    csv_path = V2 / "test_predictions.csv"
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == metadata["test_predictions_sha256"]
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    with (ROOT / "data/splits/test.csv").open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    assert len(predictions) == len(source) == 1356
    assert len({row["path"] for row in predictions}) == 1356
    assert all(pred["path"] == row["path"] and pred["true_label_index"] == row["label"]
               and pred["status"] == "ok" for pred, row in zip(predictions, source))
    assert metadata["test_prediction_errors"] == 0
    published = json.loads((V2 / "test_metrics.json").read_text(encoding="utf-8"))
    truth = np.array([int(row["true_label_index"]) for row in predictions])
    predicted = np.array([int(row["predicted_label_index"]) for row in predictions])
    recomputed = metrics(truth, predicted)
    assert all(recomputed[key] == pytest.approx(published[key]) for key in recomputed)
    assert published["correct"] == int(np.sum(truth == predicted))
    assert published["confusion_matrix"] == confusion_matrix(truth, predicted, labels=[0, 1, 2, 3]).tolist()

    # Re-run real image inference when the official dataset is available locally.
    data_root = ROOT.parent / "Project_Dudoanbenhcaycachua/data/raw"
    if data_root.is_dir():
        sample = predictions[0]
        image_path = data_root / sample["path"]
        result = predict_file(image_path, checkpoint, V2 / "model_metadata.json")
        assert result["predicted_index"] == int(sample["predicted_label_index"])
        assert len(result["decision_scores"]) == 4
        with Image.open(image_path) as image:
            png_path = tmp_path / "real_leaf.png"
            image.save(png_path)
        assert predict_file(png_path, checkpoint, V2 / "model_metadata.json")["predicted_index"] == result["predicted_index"]
        corrupt_path = tmp_path / "corrupt.jpg"
        corrupt_path.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="Cannot read image"):
            predict_file(corrupt_path, checkpoint, V2 / "model_metadata.json")


def test_v2_selection_recomputed_from_validation_log():
    with (V2 / "validation_experiments.csv").open(encoding="utf-8-sig", newline="") as handle:
        trials = list(csv.DictReader(handle))
    log = json.loads((V2 / "train_log.json").read_text(encoding="utf-8"))
    metadata = json.loads((V2 / "model_metadata.json").read_text(encoding="utf-8"))
    assert len(trials) == len(CONFIGS) * 5 * 2
    assert len(log["experiments"]) == len(trials)
    selected = max((row for row in trials if row["eligible_for_selection"] == "True"),
                   key=lambda row: (float(row["val_macro_f1"]),
                                    -float(row["val_predict_ms_per_image"]), -int(row["trial"])))
    assert int(selected["trial"]) == log["selected_trial"] == metadata["validation"]["trial"]
    assert metadata["split_csv_sha256"] == json.loads(
        (ROOT / "results/hog_svm/model_metadata.json").read_text(encoding="utf-8")
    )["split_csv_sha256"]
