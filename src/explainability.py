"""Optional Grad-CAM visualization for the two validated tomato classifiers.

This is a model-attention visualization, NOT ground-truth lesion detection.
The image rendered is the exact resized/cropped field of view used by the model.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision.transforms import CenterCrop, Resize

from src.inference import (
    ROOT, MODELS, InferenceError, _cached_model, class_names, load_config,
    load_transform, resolve_device, validate_metadata,
)


def _model_view(image: Image.Image, model_name: str, config: dict) -> Image.Image:
    """Match the spatial geometry of NV1's deterministic Val/Test transform."""
    size = config["preprocessing"]["image_size"]
    if model_name == "resnet18":
        image = Resize(config["preprocessing"]["resize_size"])(image)
        return CenterCrop(size)(image)
    return Resize((size, size))(image)


def _largest_region(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the strongest connected component, avoiding a bounding box across islands."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_count = 0
    best_box = None
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            count, x0, y0, x1, y1 = 0, x, y, x, y
            while stack:
                cy, cx = stack.pop()
                count += 1
                x0, x1 = min(x0, cx), max(x1, cx)
                y0, y1 = min(y0, cy), max(y1, cy)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if count > best_count:
                best_count, best_box = count, (x0, y0, x1, y1)
    if best_box is None or best_count < max(16, int(h * w * 0.005)):
        return None
    x0, y0, x1, y1 = best_box
    if (x1 - x0 + 1) * (y1 - y0 + 1) >= 0.85 * h * w:
        return None
    return best_box


def gradcam_map(model: torch.nn.Module, layer: torch.nn.Module, batch: torch.Tensor,
                target_index: int) -> np.ndarray | None:
    """Return [0,1] heatmap in input pixels without modifying model weights."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["activations"] = output
        output.register_hook(lambda grad: captured.__setitem__("gradients", grad.detach()))

    handle = layer.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            batch = batch.detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(batch)
            if logits.ndim != 2 or logits.shape[0] != 1 or not 0 <= target_index < logits.shape[1]:
                raise InferenceError("Grad-CAM: logits hoặc lớp mục tiêu không hợp lệ.")
            if not torch.isfinite(logits).all():
                raise InferenceError("Grad-CAM: logits không hữu hạn.")
            if int(logits.argmax(dim=1).item()) != target_index:
                raise InferenceError("Grad-CAM: lớp dự đoán không khớp kết quả gốc.")
            logits[0, target_index].backward()
            activations = captured["activations"].detach()
            gradients = captured["gradients"]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True).relu()
            cam = F.interpolate(cam, size=batch.shape[-2:], mode="bilinear",
                                align_corners=False)[0, 0].detach().cpu()
            if not torch.isfinite(cam).all():
                raise InferenceError("Grad-CAM: heatmap không hữu hạn.")
            top = float(cam.max())
            if top < 1e-10:
                return None
            cam = cam / top
            return cam.numpy()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def explain_image(image: Image.Image, model_name: str, target_index: int,
                  *, root: Path = ROOT, device: str = "cpu") -> dict:
    """Produce a PNG overlay and an approximate attention box, if supported.

    Input image must already be decoded via src.inference.read_image().
    Returns no box for Healthy because a disease box would be misleading.
    """
    if model_name not in MODELS:
        raise InferenceError(f"Grad-CAM: mô hình không hỗ trợ: {model_name}")
    names = class_names(root)
    config = load_config(model_name, root)
    validate_metadata(model_name, root, config, names)
    device = resolve_device(device)
    checkpoint = root / "checkpoints" / MODELS[model_name]
    if not checkpoint.is_file():
        raise InferenceError(f"Thiếu checkpoint {checkpoint}")
    model = _cached_model(model_name, root, device, checkpoint.stat().st_mtime_ns)
    layer = model.features[-1] if model_name == "cnn" else model.layer4[-1]
    view = _model_view(image.convert("RGB"), model_name, config)
    batch = load_transform(model_name, config)(image.convert("RGB")).unsqueeze(0).to(device)
    cam = gradcam_map(model, layer, batch, target_index)
    if cam is None:
        return {"png": None, "has_box": False, "message": "Grad-CAM không tạo được vùng chú ý ổn định."}

    base = np.asarray(view, dtype=np.float32)
    alpha = (np.clip((cam - 0.2) / 0.8, 0.0, 1.0) * 0.52)[..., None]
    red = np.array([255.0, 50.0, 35.0], dtype=np.float32)
    blended = (base * (1.0 - alpha) + red * alpha).clip(0, 255).astype(np.uint8)
    annotated = Image.fromarray(blended, mode="RGB")
    has_box = False
    if names[target_index] != "Healthy":
        region = _largest_region(cam >= 0.75)
        if region is not None:
            draw = ImageDraw.Draw(annotated)
            draw.rectangle(region, outline=(255, 0, 0), width=max(2, view.width // 100))
            has_box = True
    out = io.BytesIO()
    annotated.save(out, format="PNG")
    return {
        "png": out.getvalue(), "has_box": has_box,
        "message": "Khung đỏ: vùng chú ý xấp xỉ theo Grad-CAM; không phải vị trí bệnh được xác thực."
        if has_box else
        "Màu đỏ: vùng chú ý theo Grad-CAM. Không xác định được khung ổn định hoặc ảnh được dự đoán Healthy.",
    }
