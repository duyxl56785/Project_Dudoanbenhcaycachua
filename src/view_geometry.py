"""Where each model actually looks on the ORIGINAL uploaded image (no torch needed).

Mirrors the deterministic Val/Test geometry in src/data/transforms.py:
- CNN: Resize((size, size)) -> sees the whole image but stretches non-square photos.
- ResNet18: Resize(resize_size) on the shorter side, then CenterCrop(size) -> the
  edges of long/wide photos are never analysed.
Rounding follows torchvision: the longer side is truncated with int(), the crop
offset uses int(round(...)).
"""
from __future__ import annotations

from PIL import Image, ImageDraw

# Warn only for genuinely long/wide photos. A square photo still loses ~23% to
# ResNet18's CenterCrop, but that is the same geometry the model was trained and
# tested with (PlantVillage is 256x256), so warning there would be pure noise.
MAX_STRETCH = 1.15


def model_field_of_view(width: int, height: int, model_name: str, preprocessing: dict) -> dict:
    """Return the analysed box in original pixels, its area share and the stretch factor."""
    if width <= 0 or height <= 0:
        raise ValueError("Kích thước ảnh không hợp lệ.")
    size = int(preprocessing.get("image_size") or 224)
    stretch = max(width / height, height / width)
    if model_name != "resnet18":
        return {"box": (0.0, 0.0, float(width), float(height)), "coverage": 1.0, "stretch": stretch}
    short_target = int(preprocessing.get("resize_size") or 256)
    if width <= height:
        new_w, new_h = short_target, int(short_target * height / width)
    else:
        new_w, new_h = int(short_target * width / height), short_target
    crop_w, crop_h = min(size, new_w), min(size, new_h)
    left = int(round((new_w - crop_w) / 2.0))
    top = int(round((new_h - crop_h) / 2.0))
    sx, sy = width / new_w, height / new_h
    box = (left * sx, top * sy, (left + crop_w) * sx, (top + crop_h) * sy)
    coverage = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
    # ResNet18 keeps the aspect ratio inside the crop, so it is not stretched.
    return {"box": box, "coverage": coverage, "stretch": 1.0}


def geometry_warnings(width: int, height: int, preprocessing_by_model: dict) -> list[str]:
    """Plain-language warnings for long/wide photos the models only partially or distortedly see."""
    if max(width / height, height / width) <= MAX_STRETCH:
        return []
    messages = []
    for model_name, preprocessing in preprocessing_by_model.items():
        view = model_field_of_view(width, height, model_name, preprocessing)
        if model_name == "resnet18":
            messages.append(f"ResNet18 chỉ phân tích khoảng {view['coverage']:.0%} phần giữa ảnh "
                            "(vùng sáng trên ảnh gốc); phần bị làm tối không được phân tích.")
        if model_name == "cnn":
            messages.append(f"CNN ép cả ảnh về hình vuông nên lá bị kéo méo khoảng ×{view['stretch']:.2f}.")
    return messages


def draw_field_of_view(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """Darken everything outside the analysed box and outline it (blue, not disease red)."""
    image = image.convert("RGB")
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    dimmed = Image.blend(image, Image.new("RGB", image.size, (0, 0, 0)), 0.6)
    dimmed.paste(image.crop((x0, y0, x1, y1)), (x0, y0))
    width = max(2, min(image.size) // 150)
    ImageDraw.Draw(dimmed).rectangle((x0, y0, x1 - 1, y1 - 1), outline=(40, 120, 255), width=width)
    return dimmed
