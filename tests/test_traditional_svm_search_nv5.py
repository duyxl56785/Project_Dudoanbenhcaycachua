"""Unit/integration coverage for the NV5 traditional search/evaluation scripts."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from scripts import evaluate_best_traditional_nv5 as final_eval
from scripts import evaluate_hog_svm_experiment as hsv_eval
from scripts import search_traditional_models_nv5 as search


class TraditionalSearchTests(unittest.TestCase):
    REPO_ROOT = Path(__file__).resolve().parents[1]

    def test_feature_families_cover_hog_and_hsv_shapes(self):
        self.assertEqual(len(search.FEATURES), 5)
        self.assertIn("gray64", search.FEATURES)
        self.assertIn("rgb64_hsv", search.FEATURES)
        self.assertEqual(search.FEATURES["rgb64_hsv"]["hsv"], True)

    def test_trial_fits_scaler_and_pca_with_train_only(self):
        rng = np.random.default_rng(5)
        x_train = rng.normal(size=(40, 20)).astype(np.float32)
        y_train = np.tile(np.arange(4), 10)
        x_val = rng.normal(size=(16, 20)).astype(np.float32)
        y_val = np.tile(np.arange(4), 4)
        result = search.trial("rgb64_hsv", "hog_pca_rbf", 1.0, "scale", 8,
                              x_train, y_train, x_val, y_val)
        self.assertEqual(result["scaler_fit"], "Train only")
        self.assertEqual(result["pca_fit"], "Train only")
        self.assertTrue(0 <= result["val_macro_f1"] <= 1)

    def test_prediction_metrics_and_confusion_shape(self):
        from sklearn.metrics import confusion_matrix, f1_score
        truth = np.array([0, 1, 2, 3, 1, 2])
        pred = np.array([0, 2, 2, 3, 1, 1])
        _, macro_f1 = search.score(truth, pred)
        self.assertAlmostEqual(macro_f1, f1_score(truth, pred, labels=[0, 1, 2, 3], average="macro"))
        self.assertEqual(confusion_matrix(truth, pred, labels=[0, 1, 2, 3]).shape, (4, 4))

    def test_hsv_evaluator_refuses_to_overwrite_audited_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "artifacts"
            output.mkdir()
            (output / "metrics.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                hsv_eval.prepare_output_dir(output, overwrite=False)
            hsv_eval.prepare_output_dir(output, overwrite=True)

    def test_search_read_split_uses_manifest_and_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data/splits").mkdir(parents=True)
            split = root / "data/splits/train.csv"
            split.write_text("path,label\nraw/color/a.jpg,0\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                search.read_split(root, root / "data", "train")

    def test_final_evaluator_uses_selected_validation_config(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "best_config.json").write_text(json.dumps({"best": {"model": "hog_pca_rbf"}}), encoding="utf-8")
            with patch.object(final_eval, "read_split", side_effect=[[{"path": "a", "label": "0"}], [{"path": "b", "label": "0"}]]), \
                 patch.object(final_eval, "extract", return_value=(np.ones((1, 2), dtype=np.float32), np.array([0]))):
                # An invalid selected model must be rejected before fitting or opening Test.
                (root / "best_config.json").write_text(json.dumps({"best": {"model": "hog_linear"}}), encoding="utf-8")
                with patch("sys.argv", ["evaluate_best_traditional_nv5.py", "--data-root", td, "--search-dir", td]):
                    with self.assertRaisesRegex(ValueError, "Unexpected best model: hog_linear"):
                        final_eval.main()

    def test_recorded_validation_search_artifacts_are_consistent(self):
        """Integration check for the retained 38-trial Validation-only log."""
        search_dir = self.REPO_ROOT / "results" / "traditional_svm_search"
        with (search_dir / "validation_search.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        best_config = json.loads((search_dir / "best_config.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 38)
        self.assertEqual(best_config["trials"], len(rows))
        self.assertTrue(all(row["scaler_fit"] == "Train only" for row in rows))
        pca_rows = [row for row in rows if row["model"] == "hog_pca_rbf"]
        self.assertEqual(len(pca_rows), 2)
        self.assertEqual(best_config["best"]["model"], "hog_pca_rbf")
        self.assertAlmostEqual(best_config["best"]["val_macro_f1"], max(float(row["val_macro_f1"]) for row in rows))

    def test_primary_hsv_svm_artifacts_match_metrics_and_prediction_schema(self):
        """Integration check: the recorded primary SVM Test evidence is complete."""
        artifact_dir = self.REPO_ROOT / "results" / "hog_svm_nv5_experiment"
        metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
        with (artifact_dir / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        self.assertEqual(len(predictions), 1356)
        self.assertEqual(len({row["image_id"] for row in predictions}), 1356)
        self.assertEqual(sum(sum(row) for row in metrics["confusion_matrix"]), 1356)
        self.assertEqual(metrics["test_correct"], 1259)
        self.assertEqual(metrics["test_errors"], 97)
        self.assertAlmostEqual(metrics["accuracy"], 0.9284660766961652)
        self.assertAlmostEqual(metrics["macro_f1"], 0.9095664655252522)

    def test_primary_hsv_checkpoint_hash_and_metadata_are_consistent(self):
        """Integration check: checkpoint bytes and fitted feature dimensions agree."""
        artifact_dir = self.REPO_ROOT / "results" / "hog_svm_nv5_experiment"
        checkpoint = artifact_dir / "hog_hsv_rbf_svm.joblib"
        metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
        payload = joblib.load(checkpoint)
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), metrics["checkpoint_sha256"])
        self.assertEqual(list(payload["class_names"]), metrics["class_names"])
        self.assertEqual(payload["config"]["dimension"], 1908)
        self.assertEqual(payload["scaler"].n_features_in_, payload["config"]["dimension"])
        self.assertEqual(payload["classifier"].n_features_in_, payload["config"]["dimension"])


if __name__ == "__main__":
    unittest.main()
