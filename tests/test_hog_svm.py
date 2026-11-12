"""Regression checks for the traditional HOG/SVM pipeline."""
import csv
import hashlib
import json

import joblib
import numpy as np
import pytest
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from scripts.infer_hog_svm import predict_file
from src.hog_svm import HOGConfig, config_dict, extract_hog, image_hog, read_splits


def test_hog_shape_is_finite_and_repeatable():
    pixels = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    image = Image.fromarray(pixels)
    for config in (HOGConfig(pixels_per_cell=8), HOGConfig(pixels_per_cell=16), HOGConfig(grayscale=False)):
        first = extract_hog(image, config)
        second = extract_hog(image, config)
        assert first.shape == (config.dimension(),)
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)


def test_invalid_hog_input_and_corrupted_image(tmp_path):
    with pytest.raises(ValueError):
        HOGConfig(pixels_per_cell=10).dimension()
    with pytest.raises(ValueError):
        extract_hog("not an image", HOGConfig())
    corrupted = tmp_path / "bad.jpg"
    corrupted.write_bytes(b"not JPEG data")
    with pytest.raises(ValueError, match="Cannot read image"):
        image_hog(corrupted, HOGConfig())


def test_fixed_splits_have_no_overlap_or_missing_manifest_rows():
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[1]
    splits = read_splits(project_root, project_root / "data/raw", check_files=False)
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 4115, "val": 1031, "test": 1356}
    assert len({row["path"] for rows in splits.values() for row in rows}) == 6502


def test_group_leakage_is_rejected(tmp_path):
    (tmp_path / "data/splits").mkdir(parents=True)
    for name in ("train", "val", "test"):
        with (tmp_path / "data/splits" / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label", "group"])
            writer.writeheader()
            writer.writerow({"path": f"{name}.jpg", "label": "0", "group": "same-leaf"})
    with pytest.raises(ValueError, match="Group leakage"):
        read_splits(tmp_path, tmp_path, check_files=False)


def test_reload_and_inference_preserve_mapping_without_probabilities(tmp_path):
    config = HOGConfig(pixels_per_cell=16)
    names = ("Healthy", "Early Blight", "Late Blight", "Bacterial Spot")
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.tile(np.arange(64, dtype=np.uint8), (64, 1))).save(image_path)
    rng = np.random.default_rng(42)
    dimension = config.dimension()
    train = rng.normal(size=(20, dimension))
    labels = np.repeat(np.arange(4), 5)
    scaler = StandardScaler().fit(train)
    model = LinearSVC(C=0.1, dual=False, random_state=42).fit(scaler.transform(train), labels)
    bundle = {"hog_config": config_dict(config), "scaler": scaler, "classifier": model, "class_names": names}
    checkpoint = tmp_path / "model.joblib"
    joblib.dump(bundle, checkpoint)
    metadata_path = tmp_path / "model_metadata.json"
    metadata_path.write_text(json.dumps({"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                         "class_names": names, "hog_config": config_dict(config)}), encoding="utf-8")
    first = predict_file(image_path, checkpoint, metadata_path)
    second = predict_file(image_path, checkpoint, metadata_path)
    assert first == second
    assert first["predicted_label"] == names[first["predicted_index"]]
    assert set(first["decision_scores"]) == set(names)
    assert "probabilities" not in first
    metadata_path.write_text(json.dumps({"checkpoint_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        predict_file(image_path, checkpoint, metadata_path)


def test_published_test_predictions_match_metadata_and_fixed_split():
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[1]
    artifact_dir = project_root / "results/hog_svm"
    metadata = json.loads((artifact_dir / "model_metadata.json").read_text(encoding="utf-8"))
    csv_bytes = (artifact_dir / "test_predictions.csv").read_bytes()
    assert hashlib.sha256(csv_bytes).hexdigest() == metadata["test_predictions_sha256"]
    with (artifact_dir / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    with (project_root / "data/splits/test.csv").open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    assert len(predictions) == len(source) == 1356
    assert [row["path"] for row in predictions] == [row["path"] for row in source]
    assert all(prediction["true_label_index"] == row["label"] and prediction["status"] == "ok"
               for prediction, row in zip(predictions, source))
