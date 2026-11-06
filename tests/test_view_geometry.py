"""Field-of-view mapping back onto the original photo must match the real transforms."""
import unittest

import numpy as np
from PIL import Image

from src.view_geometry import draw_field_of_view, geometry_warnings, model_field_of_view

RESNET = {"image_size": 224, "resize_size": 256}
CNN = {"image_size": 224}
SIZES = ((256, 256), (300, 180), (180, 300), (1600, 900), (900, 1600), (4032, 3024), (257, 999))


class ViewGeometryTest(unittest.TestCase):
    def test_square_and_cnn_views(self):
        cnn = model_field_of_view(1600, 900, "cnn", CNN)
        self.assertEqual(cnn["box"], (0.0, 0.0, 1600.0, 900.0))
        self.assertEqual(cnn["coverage"], 1.0)
        self.assertAlmostEqual(cnn["stretch"], 16 / 9)
        square = model_field_of_view(512, 512, "resnet18", RESNET)
        self.assertAlmostEqual(square["coverage"], (224 / 256) ** 2)

    def test_phone_portrait_loses_more_than_half(self):
        view = model_field_of_view(900, 1600, "resnet18", RESNET)
        self.assertLess(view["coverage"], 0.45)
        x0, y0, x1, y1 = view["box"]
        self.assertAlmostEqual(x1 - x0, 900 * 224 / 256, delta=1)
        self.assertGreater(y0, 400)  # A leaf in the top quarter is outside the crop.

    def test_warnings_only_for_affected_models(self):
        # Square PlantVillage-like photos match the training geometry: no warning for either model.
        self.assertEqual(geometry_warnings(256, 256, {"cnn": CNN, "resnet18": RESNET}), [])
        self.assertEqual(geometry_warnings(1100, 1000, {"cnn": CNN, "resnet18": RESNET}), [])
        both = geometry_warnings(900, 1600, {"cnn": CNN, "resnet18": RESNET})
        self.assertEqual(len(both), 2)
        self.assertTrue(any("ResNet18" in message for message in both))
        self.assertTrue(any("CNN" in message for message in both))

    def test_invalid_size_is_rejected(self):
        with self.assertRaises(ValueError):
            model_field_of_view(0, 10, "cnn", CNN)

    def test_drawing_keeps_size_and_darkens_outside(self):
        image = Image.new("RGB", (300, 180), (200, 200, 200))
        box = model_field_of_view(300, 180, "resnet18", RESNET)["box"]
        out = np.asarray(draw_field_of_view(image, box))
        self.assertEqual(out.shape, (180, 300, 3))
        self.assertLess(out[90, 2].mean(), 100)      # Left margin is outside the crop.
        self.assertEqual(out[90, 150].tolist(), [200, 200, 200])  # Centre is untouched.

    def test_box_matches_torchvision_crop_pixels(self):
        try:
            from torchvision.transforms import CenterCrop, Resize
        except Exception as exc:  # torchvision may be unavailable in light environments.
            self.skipTest(f"torchvision unavailable: {exc}")
        for width, height in SIZES:
            with self.subTest(size=(width, height)):
                # Smooth position-encoding image: any misplaced crop shows up as a colour shift.
                xs, ys = np.meshgrid(np.linspace(0, 255, width), np.linspace(0, 255, height))
                image = Image.fromarray(np.stack([xs, ys, 255 - xs], axis=-1).astype(np.uint8))
                expected = CenterCrop(224)(Resize(256)(image))
                x0, y0, x1, y1 = model_field_of_view(width, height, "resnet18", RESNET)["box"]
                # Cropping the original at the mapped box then resizing must reproduce the model view.
                mapped = image.resize((224, 224), Image.BILINEAR, box=(x0, y0, x1, y1))
                diff = np.abs(np.asarray(mapped, dtype=float) - np.asarray(expected, dtype=float)).mean()
                self.assertLess(diff, 2.0)
                resized_w, resized_h = Resize(256)(image).size
                self.assertAlmostEqual((x1 - x0) * resized_w / width, 224, delta=1e-6)
                self.assertAlmostEqual((y1 - y0) * resized_h / height, 224, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
