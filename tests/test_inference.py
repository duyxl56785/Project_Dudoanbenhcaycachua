import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.inference import InferenceError, MAX_IMAGE_BYTES, class_names, load_config, load_model, read_image


class InferenceTests(unittest.TestCase):
    def test_mapping_preserves_numeric_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/class_mapping.json").write_text(json.dumps({"0": "Z", "1": "A", "2": "Y", "3": "B"}))
            self.assertEqual(class_names(root), ("Z", "A", "Y", "B"))
            (root / "configs/class_mapping.json").write_text(json.dumps({"0": "Z", "1": "A"}))
            with self.assertRaises(InferenceError):
                class_names(root)

    def test_image_validation(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (3, 3), "red").save(buffer, format="PNG")
        self.assertEqual(read_image(buffer.getvalue()).mode, "RGB")
        for data in (b"", b"broken", b"x" * (MAX_IMAGE_BYTES + 1), buffer.getvalue()[:20]):
            with self.subTest(size=len(data)), self.assertRaises(InferenceError):
                read_image(data)

    def test_unfinished_config_and_missing_checkpoint(self):
        with self.assertRaisesRegex(InferenceError, "preprocessing chưa chốt"):
            load_config("cnn")
        with self.assertRaisesRegex(InferenceError, "Thiếu checkpoint"):
            load_model("cnn")


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch chưa cài; test API tensor chờ môi trường có torch")
class TensorTests(unittest.TestCase):
    def test_shape_probabilities_and_top_label(self):
        from src.inference import predict

        class Tiny(torch.nn.Module):
            def forward(self, batch):
                return torch.tensor([[0.0, 1.0, 4.0, 2.0]])

        names = ("Z", "A", "Y", "B")
        result = predict(None, Tiny(), lambda _: torch.zeros(3, 2, 2), names)
        self.assertEqual(result["top_label"], "Y")
        self.assertEqual(tuple(result["probabilities"]), names)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1)

    def test_wrong_output_shape(self):
        from src.inference import predict

        class Bad(torch.nn.Module):
            def forward(self, batch):
                return torch.zeros(1, 3)

        with self.assertRaisesRegex(InferenceError, "shape"):
            predict(None, Bad(), lambda _: torch.zeros(3, 2, 2), ("a", "b", "c", "d"))

    def test_incompatible_checkpoint(self):
        from src.inference import load_model

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoints").mkdir()
            torch.save({"wrong": torch.zeros(1)}, root / "checkpoints/cnn_best.pt")
            with self.assertRaisesRegex(InferenceError, "Không tải được checkpoint"):
                load_model("cnn", root, factory=lambda num_classes: torch.nn.Linear(2, num_classes))


if __name__ == "__main__":
    unittest.main()
