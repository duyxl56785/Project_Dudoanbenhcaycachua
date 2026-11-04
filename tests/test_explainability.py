"""Regression tests for experimental Grad-CAM overlays (no real checkpoints required)."""
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from src.explainability import _largest_region, _model_view, gradcam_map
from src.inference import InferenceError


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(0.5)
            self.fc.weight[0].fill_(1.0)
            self.fc.weight[1].fill_(0.0)

    def forward(self, image):
        activations = torch.relu(self.conv(image))
        return self.fc(self.pool(activations).flatten(1))


class GradCamTests(unittest.TestCase):
    def test_heatmap_follows_salient_area_without_modifying_weights(self):
        model = TinyClassifier().eval()
        image = torch.zeros(1, 3, 32, 32)
        image[:, :, 5:11, 7:16] = 1.0
        before = [param.detach().clone() for param in model.parameters()]
        cam = gradcam_map(model, model.conv, image, 0)
        self.assertIsNotNone(cam)
        self.assertEqual(cam.shape, (32, 32))
        self.assertTrue(np.isfinite(cam).all())
        self.assertGreater(float(cam[7, 10]), 0.95)
        self.assertLess(float(cam[25, 25]), 0.05)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_wrong_target_is_rejected_and_hook_is_cleaned(self):
        model = TinyClassifier().eval()
        image = torch.ones(1, 3, 32, 32)
        with self.assertRaisesRegex(InferenceError, "không khớp"):
            gradcam_map(model, model.conv, image, 1)
        self.assertEqual(len(model.conv._forward_hooks), 0)

    def test_approximate_box_prefers_largest_connected_region(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[2:8, 2:8] = True
        mask[12:25, 15:29] = True
        self.assertEqual(_largest_region(mask), (15, 12, 28, 24))
        self.assertIsNone(_largest_region(np.zeros((32, 32), dtype=bool)))

    def test_model_view_is_model_specific(self):
        image = Image.new("RGB", (400, 300), "green")
        self.assertEqual(_model_view(image, "cnn", {"preprocessing": {"image_size": 224}}).size, (224, 224))
        self.assertEqual(
            _model_view(image, "resnet18", {"preprocessing": {"image_size": 224, "resize_size": 256}}).size,
            (224, 224),
        )


if __name__ == "__main__":
    unittest.main()
