import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_models import load_class_mapping, load_predictions, metrics_from_json, run, validate_same_test_set

MAPPING = {0: "Healthy", 1: "Early Blight", 2: "Late Blight", 3: "Bacterial Spot"}


def write_predictions(path, rows, split="test"):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "true", "pred", "split"])
        writer.writeheader()
        writer.writerows({**row, "split": split} for row in rows)


def write_metrics(path, n_samples=4):
    path.write_text(json.dumps({
        "split": "test", "n_samples": n_samples, "accuracy": 0.75,
        "macro_precision": 0.75, "macro_recall": 0.75, "macro_f1": 0.75,
        "per_class": {name: {"precision": 0.75, "recall": 0.75, "f1": 0.75, "support": 1}
                      for name in MAPPING.values()},
        "confusion_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }), encoding="utf-8")


class EvaluateModelsTests(unittest.TestCase):
    def test_mapping_is_contiguous_and_rejects_gap(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as directory:
            root = Path(directory)
            good = root / "mapping.json"
            good.write_text(json.dumps({"0": "Healthy", "1": "Sick"}), encoding="utf-8")
            self.assertEqual(load_class_mapping(good), {0: "Healthy", 1: "Sick"})
            bad = root / "bad.json"
            bad.write_text(json.dumps({"0": "Healthy", "2": "Sick"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "liên tục"):
                load_class_mapping(bad)

    def test_prediction_schema_and_split_validation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as directory:
            path = Path(directory) / "pred.csv"
            write_predictions(path, [{"path": "a", "true": 0, "pred": 0}], split="val")
            with self.assertRaisesRegex(ValueError, "không khớp"):
                load_predictions(path, MAPPING, expected_split="test")
            bad = Path(directory) / "bad.csv"
            bad.write_text("path,true\na,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prediction"):
                load_predictions(bad, MAPPING)

    def test_same_test_set_checks_ids_and_ground_truth(self):
        first = [{"id": "a", "true": 0}, {"id": "b", "true": 1}]
        validate_same_test_set(first, [{"id": "a", "true": 0}, {"id": "b", "true": 1}])
        with self.assertRaisesRegex(ValueError, "cùng Test"):
            validate_same_test_set(first, [{"id": "a", "true": 0}])
        with self.assertRaisesRegex(ValueError, "Ground truth"):
            validate_same_test_set(first, [{"id": "a", "true": 0}, {"id": "b", "true": 2}])

    def test_metrics_json_schema_and_missing_fields(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as directory:
            metrics = Path(directory) / "metrics.json"
            write_metrics(metrics)
            loaded = metrics_from_json(metrics, MAPPING)
            self.assertEqual(loaded["macro_f1"], 0.75)
            self.assertEqual(loaded["per_class"]["Healthy"]["f1"], 0.75)
            incomplete = Path(directory) / "incomplete.json"
            incomplete.write_text(json.dumps({"accuracy": 0.5}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "thiếu trường"):
                metrics_from_json(incomplete, MAPPING)

    def test_prediction_and_reported_metrics_must_match(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as directory:
            root = Path(directory)
            predictions = root / "cnn_test_predictions.csv"
            write_predictions(predictions, [
                {"path": "a", "true": 0, "pred": 0},
                {"path": "b", "true": 1, "pred": 2},
            ])
            metrics = root / "cnn_metrics.json"
            write_metrics(metrics, n_samples=2)
            # The fixture reports 0.75, while the two rows have accuracy 0.5.
            args = type("Args", (), {
                "class_mapping": str(Path(__file__).parents[1] / "configs" / "class_mapping.json"),
                "cnn_predictions": str(predictions), "cnn_metrics": str(metrics),
                "resnet18_predictions": None, "resnet18_metrics": None,
                "split": "test", "output_dir": str(root / "results"),
            })()
            with self.assertRaisesRegex(ValueError, "không khớp"):
                run(args)

    def test_run_writes_real_cnn_and_pending_resnet(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as directory:
            root = Path(directory); metrics = root / "cnn_metrics.json"; write_metrics(metrics)
            output = root / "results"
            args = type("Args", (), {
                "class_mapping": str(Path(__file__).parents[1] / "configs" / "class_mapping.json"),
                "cnn_predictions": None, "cnn_metrics": str(metrics), "resnet18_predictions": None,
                "resnet18_metrics": None, "split": "test", "output_dir": str(output),
            })()
            with patch("scripts.evaluate_models.plot_confusion"):
                result = run(args)
            self.assertEqual(result["models"], {"CNN": "computed", "ResNet18": "pending"})
            comparison = (output / "comparison.csv").read_text(encoding="utf-8")
            self.assertIn("CNN,computed", comparison)
            self.assertIn("ResNet18,pending", comparison)
            self.assertIn("ResNet18", (output / "analysis.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
