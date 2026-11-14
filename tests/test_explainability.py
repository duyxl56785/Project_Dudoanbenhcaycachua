"""Regression tests for experimental Grad-CAM overlays (no real checkpoints required)."""
import unittest
import io
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
from PIL import Image
import torch
from torch import nn

from src.explainability import _largest_region, _model_view, explain_image, gradcam_map
from src.inference import InferenceError, load_config, load_transform, predict


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
        model = TinyClassifier().train()
        image = torch.ones(1, 3, 32, 32)
        with self.assertRaisesRegex(InferenceError, "không khớp"):
            gradcam_map(model, model.conv, image, 1)
        self.assertEqual(len(model.conv._forward_hooks), 0)
        self.assertTrue(model.training)

    def test_preserves_existing_gradients_modes_and_batchnorm_buffers(self):
        from src.models.cnn import build_model
        model = build_model().train()
        model.dropout.eval()
        modes = [module.training for module in model.modules()]
        state = {name: value.clone() for name, value in model.state_dict().items()}
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        gradients = [parameter.grad for parameter in model.parameters()]
        image = torch.randn(1, 3, 32, 32)
        model.eval()
        with torch.no_grad():
            target = int(model(image).argmax())
        for module, training in zip(model.modules(), modes):
            module.training = training
        gradcam_map(model, model.features[-1], image, target)
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertTrue(all(torch.equal(state[name], value) for name, value in model.state_dict().items()))
        self.assertTrue(all(p.grad is g and torch.all(g == 1) for p, g in zip(model.parameters(), gradients)))
        self.assertEqual(len(model.features[-1]._forward_hooks), 0)

    def test_uses_logit_gradient_instead_of_softmax(self):
        model = TinyClassifier().eval()
        with torch.no_grad():
            model.conv.weight.zero_()
            model.conv.weight[0, 0] = 1
            model.conv.weight[1, 1] = 1
            model.fc.weight[0] = torch.tensor([2., 1.])
            model.fc.weight[1] = torch.tensor([0., 2.])
        image = torch.zeros(1, 3, 8, 8)
        image[:, 0, :4] = 1
        image[:, 1, 4:] = 1
        cam = gradcam_map(model, model.conv, image, 0)
        np.testing.assert_allclose(cam[:4], 1.)
        np.testing.assert_allclose(cam[4:], 0.5)

    def test_concurrent_prediction_waits_for_attention_forward(self):
        model = TinyClassifier().eval()
        image = torch.ones(1, 3, 8, 8)
        entered, release, started, finished = (Event() for _ in range(4))
        def pause_gradient_forward(*_):
            if torch.is_grad_enabled():
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("Timed out waiting for concurrent prediction")
        def classify():
            started.set()
            result = predict(None, model, lambda _: image[0], ("a", "b"))
            finished.set()
            return result
        handle = model.conv.register_forward_hook(pause_gradient_forward)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                cam = pool.submit(gradcam_map, model, model.conv, image, 0)
                try:
                    self.assertTrue(entered.wait(5))
                    classification = pool.submit(classify)
                    self.assertTrue(started.wait(5))
                    self.assertFalse(finished.wait(0.1))
                finally:
                    release.set()
                self.assertIsNotNone(cam.result(timeout=5))
                self.assertEqual(classification.result(timeout=5)["top_index"], 0)
        finally:
            handle.remove()
        self.assertEqual(len(model.conv._forward_hooks), 0)

    def test_frozen_models_and_inference_context(self):
        from src.models.cnn import build_model
        from src.models.resnet18 import build_model as resnet18
        for factory, layer_name in ((build_model, "features"), (resnet18, "layer4")):
            with self.subTest(model=layer_name):
                model = factory().eval().requires_grad_(False)
                layer = model.features[-1] if layer_name == "features" else model.layer4
                with torch.inference_mode():
                    image = torch.randn(1, 3, 32, 32)
                    target = int(model(image).argmax())
                    cam = gradcam_map(model, layer, image, target)
                self.assertTrue(cam is None or (cam.shape == (32, 32) and np.isfinite(cam).all()))
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
                self.assertEqual(len(layer._forward_hooks), 0)

    def test_flat_zero_tiny_and_diffuse_regions_have_no_box(self):
        model = TinyClassifier().eval()
        self.assertIsNone(gradcam_map(model, model.conv, torch.zeros(1, 3, 8, 8), 0))
        self.assertIsNone(_largest_region(np.ones((32, 32), dtype=bool)))
        mask = np.zeros((32, 32), dtype=bool)
        mask[2:4, 2:4] = True
        self.assertIsNone(_largest_region(mask))

    def test_overlay_skips_healthy_box_and_empty_heatmap(self):
        cam = np.zeros((224, 224), dtype=np.float32)
        cam[40:100, 60:120] = 1
        image = Image.new("RGB", (400, 300), "green")
        with patch("src.explainability._cached_model"), patch("src.explainability.validate_metadata"), \
             patch("src.explainability.Path.is_file", return_value=True), patch("src.explainability.Path.stat"), \
             patch("src.explainability.gradcam_map", return_value=cam) as make_cam:
            for model in ("cnn", "resnet18"):
                healthy = explain_image(image, model, 0)
                disease = explain_image(image, model, 1)
                self.assertFalse(healthy["has_box"])
                self.assertTrue(disease["has_box"])
                with Image.open(io.BytesIO(disease["png"])) as overlay:
                    self.assertEqual(overlay.size, (224, 224))
                make_cam.return_value = None
                self.assertIsNone(explain_image(image, model, 1)["png"])
                make_cam.return_value = cam

    def test_model_view_matches_preprocessed_pixels_for_non_square_images(self):
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (137, 509, 3), dtype=np.uint8))
        for model in ("cnn", "resnet18"):
            config = load_config(model)
            prep = config["preprocessing"]
            tensor = load_transform(model, config)(image)
            restored = tensor * torch.tensor(prep["normalize_std"])[:, None, None]
            restored += torch.tensor(prep["normalize_mean"])[:, None, None]
            np.testing.assert_allclose(restored.permute(1, 2, 0).numpy() * 255,
                                       np.asarray(_model_view(image, model, config)), atol=0.0001)

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
