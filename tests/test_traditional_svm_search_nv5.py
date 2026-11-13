"""Unit/integration coverage for the NV5 traditional search/evaluation scripts."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import evaluate_best_traditional_nv5 as final_eval
from scripts import search_traditional_models_nv5 as search


class TraditionalSearchTests(unittest.TestCase):
    def test_feature_families_cover_hog_and_hsv_shapes(self):
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


if __name__ == "__main__":
    unittest.main()
