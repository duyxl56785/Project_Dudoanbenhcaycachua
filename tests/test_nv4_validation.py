import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.inference import (
    InferenceError,
    MAX_IMAGE_BYTES,
    class_names,
    load_model,
    predict,
    read_image,
    validate_metadata,
)


def image_bytes(mode="RGB", image_format="PNG", size=(17, 29)):
    buffer = io.BytesIO()
    color = 128 if mode == "L" else (255, 0, 0, 128) if mode == "RGBA" else "red"
    Image.new(mode, size, color).save(buffer, format=image_format)
    return buffer.getvalue()


class ImageInputValidationTests(unittest.TestCase):
    def test_accepts_jpg_jpeg_and_png_content(self):
        for image_format in ("JPEG", "PNG"):
            with self.subTest(image_format=image_format):
                decoded = read_image(image_bytes("RGB", image_format))
                self.assertEqual(decoded.mode, "RGB")
                self.assertEqual(decoded.size, (17, 29))

    def test_converts_grayscale_and_rgba_to_rgb(self):
        for mode in ("L", "RGBA"):
            with self.subTest(mode=mode):
                decoded = read_image(image_bytes(mode))
                self.assertEqual(decoded.mode, "RGB")
                self.assertEqual(decoded.size, (17, 29))

    def test_accepts_non_256_dimensions(self):
        decoded = read_image(image_bytes(size=(91, 37)))
        self.assertEqual(decoded.size, (91, 37))

    def test_rejects_empty_corrupt_truncated_and_oversized_content(self):
        valid = image_bytes()
        cases = {
            "empty": b"",
            "txt_renamed_jpg": b"not an image",
            "corrupt": b"\x89PNG\r\n\x1a\ncorrupt",
            "truncated": valid[:24],
            "oversized": b"x" * (MAX_IMAGE_BYTES + 1),
        }
        for name, payload in cases.items():
            with self.subTest(name=name), self.assertRaises(InferenceError):
                read_image(payload)

    def test_rejects_unsupported_image_content_even_if_decodable(self):
        with self.assertRaisesRegex(InferenceError, "Chỉ nhận nội dung ảnh JPEG hoặc PNG"):
            read_image(image_bytes("RGB", "GIF"))


class MappingValidationTests(unittest.TestCase):
    def test_requires_exactly_four_unique_labels_in_numeric_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            mapping = root / "configs/class_mapping.json"
            mapping.write_text(json.dumps({"0": "Healthy", "1": "Early Blight", "2": "Late Blight", "3": "Bacterial Spot"}), encoding="utf-8")
            self.assertEqual(class_names(root), ("Healthy", "Early Blight", "Late Blight", "Bacterial Spot"))
            mapping.write_text(json.dumps({"0": "Healthy", "1": "Healthy", "2": "Late Blight", "3": "Bacterial Spot"}), encoding="utf-8")
            with self.assertRaises(InferenceError):
                class_names(root)

    def test_metadata_must_match_labels_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoints").mkdir()
            names = ("Healthy", "Early Blight", "Late Blight", "Bacterial Spot")
            config = {"preprocessing": {"image_size": 224, "normalize_mean": [0.1, 0.2, 0.3], "normalize_std": [1, 1, 1]}}
            metadata = root / "checkpoints/cnn_best.meta.json"
            metadata.write_text(json.dumps({"labels": list(names), "preprocessing": {"image_size": 128}}), encoding="utf-8")
            with self.assertRaisesRegex(InferenceError, "preprocessing khác"):
                validate_metadata("cnn", root, config, names)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch chưa cài")
class CheckpointAndInferenceTests(unittest.TestCase):
    def _root_with_checkpoint(self, payload):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "checkpoints").mkdir()
        torch.save(payload, root / "checkpoints/cnn_best.pt")
        return temporary, root

    def test_loads_plain_state_dict_on_cpu_and_sets_eval(self):
        source = torch.nn.Linear(2, 4)
        temporary, root = self._root_with_checkpoint(source.state_dict())
        with temporary:
            loaded = load_model("cnn", root, device="cpu", factory=lambda num_classes: torch.nn.Linear(2, num_classes))
            self.assertFalse(loaded.training)
            self.assertEqual(next(loaded.parameters()).device.type, "cpu")

    def test_rejects_corrupt_checkpoint_without_leaking_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoints").mkdir()
            (root / "checkpoints/cnn_best.pt").write_bytes(b"corrupt checkpoint")
            with self.assertRaisesRegex(InferenceError, "Không tải được checkpoint"):
                load_model("cnn", root, factory=lambda num_classes: torch.nn.Linear(2, num_classes))

    def test_rejects_non_state_dict_and_wrong_architecture(self):
        for payload in ([1, 2, 3], {"unexpected.weight": torch.zeros(1)}):
            temporary, root = self._root_with_checkpoint(payload)
            with temporary, self.subTest(payload_type=type(payload).__name__), self.assertRaises(InferenceError):
                load_model("cnn", root, factory=lambda num_classes: torch.nn.Linear(2, num_classes))

    def test_input_output_shape_softmax_and_top1(self):
        seen = {}

        class Model(torch.nn.Module):
            def forward(self, batch):
                seen["shape"] = tuple(batch.shape)
                return torch.tensor([[0.0, 2.0, 1.0, -1.0]])

        names = ("Healthy", "Early Blight", "Late Blight", "Bacterial Spot")
        result = predict(Image.new("RGB", (256, 256)), Model(), lambda _: torch.zeros(3, 224, 224), names, "cpu")
        self.assertEqual(seen["shape"], (1, 3, 224, 224))
        self.assertEqual(result["top_label"], "Early Blight")
        self.assertEqual(tuple(result["probabilities"]), names)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=6)
        self.assertEqual(result["top_index"], max(range(4), key=lambda index: tuple(result["probabilities"].values())[index]))

    def test_rejects_nan_logits_and_wrong_class_count(self):
        names = ("a", "b", "c", "d")
        for logits in (torch.tensor([[0.0, float("nan"), 1.0, 2.0]]), torch.zeros(1, 3)):
            class Model(torch.nn.Module):
                def forward(self, batch, output=logits):
                    return output

            with self.subTest(shape=tuple(logits.shape)), self.assertRaises(InferenceError):
                predict(None, Model(), lambda _: torch.zeros(3, 224, 224), names, "cpu")


if __name__ == "__main__":
    unittest.main()
