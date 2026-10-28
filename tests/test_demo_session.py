import csv
import io
import unittest
from datetime import datetime, timezone

from PIL import Image

from demo.session import MAX_UPLOADS, dashboard, export_csv, predict_uploads, selected_models
from src.inference import InferenceError


class Upload:
    def __init__(self, name, data):
        self.name, self.data = name, data

    def getvalue(self):
        return self.data


def picture():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "green").save(buffer, format="JPEG")
    return buffer.getvalue()


class SessionTests(unittest.TestCase):
    def test_batch_keeps_results_associated_and_continues_after_bad_image(self):
        uploads = [Upload("a.jpg", picture()), Upload("bad.jpg", b"broken"), Upload("b.jpg", picture())]
        labels = iter(("Healthy", "Late Blight"))

        def predict(image, model, device):
            label = next(labels)
            return {"top_label": label, "probabilities": {"Healthy": 0.8 if label == "Healthy" else 0.2,
                    "Early Blight": 0.0, "Late Blight": 0.2 if label == "Healthy" else 0.8,
                    "Bacterial Spot": 0.0}, "forward_ms": 4.0}

        fixed = lambda: datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
        items, records = predict_uploads(uploads, ("cnn",), predict, device="cpu", now=fixed)
        self.assertEqual([item["name"] for item in items], ["a.jpg", "bad.jpg", "b.jpg"])
        self.assertIn("image", items[1]["errors"])
        self.assertEqual([row["top_label"] for row in records], ["Healthy", "Late Blight"])
        self.assertEqual([row["image_name"] for row in records], ["a.jpg", "b.jpg"])
        self.assertNotIn("image", records[0])
        self.assertEqual(dashboard(records, tuple(records[0]["probabilities"]))["predictions"], 2)

    def test_missing_model_does_not_hide_cnn(self):
        def predict(image, model, device):
            if model == "resnet18":
                raise InferenceError("Thiếu checkpoint")
            return {"top_label": "Healthy", "probabilities": {"Healthy": 1.0, "Early Blight": 0.0,
                    "Late Blight": 0.0, "Bacterial Spot": 0.0}, "forward_ms": 3.0}

        items, records = predict_uploads([Upload("a.jpg", picture())], selected_models("So sánh cả hai"), predict)
        self.assertEqual(len(records), 1)
        self.assertIn("cnn", items[0]["results"])
        self.assertEqual(items[0]["errors"]["resnet18"], "Thiếu checkpoint")

    def test_limit_and_csv_excel_encoding(self):
        with self.assertRaisesRegex(InferenceError, str(MAX_UPLOADS)):
            predict_uploads([Upload("x.jpg", b"")] * (MAX_UPLOADS + 1), ("cnn",), lambda *_: None)
        record = {"image_name": "lá cà chua.jpg", "model": "CNN", "top_label": "Healthy", "top_probability": 1.0,
                  "probabilities": {"Healthy": 1.0, "Early Blight": 0.0, "Late Blight": 0.0, "Bacterial Spot": 0.0},
                  "predicted_at": "2026-10-28T12:00:00+00:00", "forward_ms": 4.5}
        content = export_csv([record], tuple(record["probabilities"]))
        self.assertTrue(content.startswith(b"\xef\xbb\xbf"))
        rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
        self.assertEqual(rows[1][0:4], ["lá cà chua.jpg", "CNN", "Healthy", "1.000000"])
        self.assertEqual(len(rows[0]), 10)

    def test_dashboard_empty_and_session_aggregates(self):
        names = ("Healthy", "Early Blight", "Late Blight", "Bacterial Spot")
        self.assertIsNone(dashboard([], names)["mean_forward_ms"])
        records = [{"image_name": "a.jpg", "model": "CNN", "top_label": "Healthy", "forward_ms": 2.0},
                   {"image_name": "a.jpg", "model": "ResNet18", "top_label": "Late Blight", "forward_ms": 4.0}]
        summary = dashboard(records, names)
        self.assertEqual((summary["images"], summary["predictions"], summary["mean_forward_ms"]), (1, 2, 3.0))
        self.assertEqual(summary["by_model"], {"CNN": 1, "ResNet18": 1})
