"""Check the evaluation's safeguards around published Test predictions."""
import pytest

from scripts.evaluate_hog_svm import validate_test_rows


def test_test_evaluation_rejects_misaligned_paths_or_labels():
    names = ("Healthy", "Early Blight", "Late Blight", "Bacterial Spot")
    source = [{"path": "raw/color/a.jpg", "label": "0"}]
    row = {"path": source[0]["path"], "true_label_index": "0", "true_label": names[0],
           "predicted_label_index": "1", "predicted_label": names[1], "status": "ok",
           **{f"decision_score_{i}": "0.1" for i in range(4)}}
    validate_test_rows([row], source, names)
    with pytest.raises(ValueError, match="fixed split"):
        validate_test_rows([dict(row, path="different.jpg")], source, names)
    with pytest.raises(ValueError, match="fixed split"):
        validate_test_rows([dict(row, true_label_index="3")], source, names)
    with pytest.raises(ValueError, match="Non-finite"):
        validate_test_rows([dict(row, decision_score_0="nan")], source, names)


def test_published_comparison_is_test_only_and_has_three_models():
    import csv
    from pathlib import Path
    report = Path(__file__).resolve().parents[1] / "results/hog_svm/evaluation_20261112/comparison_three_models.csv"
    with report.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["model"] for row in rows] == ["CNN", "ResNet18", "HOG + Linear SVM"]
    assert all(row["split"] == "test" and row["n_samples"] == "1356" for row in rows)
    assert rows[1]["train_seconds"] == ""  # NV3 has no comparable duration in the handoff artifact.
