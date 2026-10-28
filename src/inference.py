"""Shared inference API. Models export build_model(num_classes); NV1 exports
build_val_transform(model_name, config). Checkpoints are plain state_dicts.
"""
import io
import json
import math
import time
from functools import lru_cache
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MODELS = {"cnn": "cnn_best.pt", "resnet18": "resnet18_best.pt"}


class InferenceError(Exception):
    pass


def class_names(root=ROOT):
    try:
        mapping = json.loads((root / "configs/class_mapping.json").read_text(encoding="utf-8"))
        if set(mapping) != {str(i) for i in range(4)}:
            raise ValueError("mapping cần đúng chỉ số 0, 1, 2, 3")
        names = tuple(mapping[str(i)] for i in range(4))
        if any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != 4:
            raise ValueError("tên lớp phải là bốn chuỗi khác nhau")
        return names
    except (OSError, ValueError, TypeError) as exc:
        raise InferenceError(f"Class mapping không hợp lệ: {exc}") from exc


def read_image(data):
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise InferenceError("Ảnh trống hoặc vượt giới hạn 10 MiB.")
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise InferenceError(f"Thiếu Pillow: {exc}") from exc
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in ("JPEG", "PNG"):
                raise InferenceError("Chỉ nhận nội dung ảnh JPEG hoặc PNG.")
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            return source.convert("RGB")
    except InferenceError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise InferenceError(f"Không đọc được ảnh JPEG/PNG: {exc}") from exc


def load_config(model_name, root=ROOT):
    if model_name not in MODELS:
        raise InferenceError(f"Mô hình không hỗ trợ: {model_name}")
    try:
        import yaml
        config = yaml.safe_load((root / f"configs/{model_name}.yaml").read_text(encoding="utf-8"))
        prep = config["preprocessing"]
        required = ("image_size", "normalize_mean", "normalize_std")
        if model_name == "resnet18":
            required += ("resize_size",)
        missing = [k for k in required if prep.get(k) is None]
        if missing:
            raise InferenceError(f"{model_name}: preprocessing chưa chốt: {', '.join(missing)}.")
        if config["model"]["num_classes"] != 4:
            raise InferenceError(f"{model_name}: num_classes phải bằng 4.")
        if not isinstance(prep["image_size"], int) or prep["image_size"] <= 0:
            raise InferenceError(f"{model_name}: image_size không hợp lệ.")
        if model_name == "resnet18" and (not isinstance(prep["resize_size"], int) or prep["resize_size"] < prep["image_size"]):
            raise InferenceError(f"{model_name}: resize_size không hợp lệ.")
        for key in ("normalize_mean", "normalize_std"):
            values = prep[key]
            if not isinstance(values, (list, tuple)) or len(values) != 3 or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
                raise InferenceError(f"{model_name}: {key} phải có ba số hữu hạn.")
        if any(v <= 0 for v in prep["normalize_std"]):
            raise InferenceError(f"{model_name}: normalize_std phải dương.")
        return config
    except InferenceError:
        raise
    except (ImportError, OSError, KeyError, TypeError, ValueError) as exc:
        raise InferenceError(f"Không đọc được config {model_name}: {exc}") from exc


def load_model(model_name, root=ROOT, device="cpu", *, factory=None):
    if model_name not in MODELS:
        raise InferenceError(f"Mô hình không hỗ trợ: {model_name}")
    path = root / "checkpoints" / MODELS[model_name]
    if not path.is_file():
        raise InferenceError(f"Thiếu checkpoint {path}; chỉ dùng file do nhóm xác nhận.")
    try:
        import torch
        if factory is None:
            module = import_module(f"src.models.{model_name}")
            factory = getattr(module, "build_model", None)
            if factory is None and model_name == "resnet18" and hasattr(module, "build_resnet18"):
                # Adapter for NV3 PR #7; inference must never fetch pretrained weights.
                factory = lambda num_classes: module.build_resnet18(num_classes=num_classes, pretrained=False)
            if factory is None:
                raise InferenceError(f"Chưa bàn giao build_model(num_classes) trong src.models.{model_name}.")
        model = factory(num_classes=4)
        state = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise InferenceError(f"Checkpoint {path} phải là state_dict tensor thuần.")
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()
    except InferenceError:
        raise
    except (ImportError, AttributeError, RuntimeError, ValueError, TypeError, OSError) as exc:
        raise InferenceError(f"Không tải được checkpoint {path}: {exc}") from exc


def validate_metadata(model_name, root, config, names):
    """Reject a supplied sidecar if its labels/preprocessing disagree with the app."""
    path = (root / "checkpoints" / MODELS[model_name]).with_suffix(".meta.json")
    if not path.is_file():
        return
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("labels") != list(names):
            raise InferenceError(f"Metadata {path}: thứ tự nhãn khác class_mapping.json.")
        if meta.get("preprocessing") != config["preprocessing"]:
            raise InferenceError(f"Metadata {path}: preprocessing khác config hiện tại.")
    except InferenceError:
        raise
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise InferenceError(f"Metadata {path} không hợp lệ: {exc}") from exc


@lru_cache(maxsize=4)
def _cached_model(model_name, root, device, modified_ns):
    """Reuse loaded weights; modification time invalidates the cache."""
    return load_model(model_name, root, device)


def resolve_device(device="auto"):
    if device == "cpu":
        return "cpu"
    if device not in ("auto", "cuda"):
        raise InferenceError(f"Thiết bị không hỗ trợ: {device}")
    try:
        import torch
        available = torch.cuda.is_available()
    except ImportError as exc:
        raise InferenceError(f"Thiếu PyTorch: {exc}") from exc
    if device == "cuda" and not available:
        raise InferenceError("CUDA chưa khả dụng trên máy này.")
    return "cuda" if available else "cpu"


def load_transform(model_name, config):
    try:
        factory = getattr(import_module("src.data.transforms"), "build_val_transform", None)
        if factory is None:
            raise InferenceError("Chưa bàn giao build_val_transform(model_name, config) trong src.data.transforms.")
        return factory(model_name, config)
    except InferenceError:
        raise
    except (ImportError, AttributeError, ValueError, TypeError) as exc:
        raise InferenceError(f"Không tải được Val/Test transform: {exc}") from exc


def predict(image, model, transform, names, device="cpu"):
    """Elapsed time includes forward pass only; CUDA is synchronized."""
    try:
        import torch
        batch = transform(image).unsqueeze(0).to(device)
        model.eval()
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(batch)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if logits.shape != (1, len(names)) or not torch.isfinite(logits).all():
            raise InferenceError(f"Đầu ra cần shape (1, {len(names)}) và giá trị hữu hạn.")
        values = torch.softmax(logits, dim=1)[0].cpu().tolist()
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in values) or not math.isclose(sum(values), 1, abs_tol=1e-5):
            raise InferenceError("Xác suất đầu ra không hợp lệ.")
        index = max(range(len(values)), key=values.__getitem__)
        return {"top_index": index, "top_label": names[index], "probabilities": dict(zip(names, values)), "forward_ms": elapsed_ms}
    except InferenceError:
        raise
    except (ImportError, AttributeError, RuntimeError, ValueError, TypeError) as exc:
        raise InferenceError(f"Không thể suy luận: {exc}") from exc


def predict_image(image, model_name, root=ROOT, device="cpu"):
    if model_name not in MODELS:
        raise InferenceError(f"Mô hình không hỗ trợ: {model_name}")
    device = resolve_device(device) if device != "cpu" else "cpu"
    names = class_names(root)
    config = load_config(model_name, root)
    validate_metadata(model_name, root, config, names)
    transform = load_transform(model_name, config)
    checkpoint = root / "checkpoints" / MODELS[model_name]
    if not checkpoint.is_file():
        raise InferenceError(f"Thiếu checkpoint {checkpoint}; chỉ dùng file do nhóm xác nhận.")
    try:
        modified_ns = checkpoint.stat().st_mtime_ns
    except OSError as exc:
        raise InferenceError(f"Không đọc được checkpoint {checkpoint}: {exc}") from exc
    model = _cached_model(model_name, root, device, modified_ns)
    return predict(image, model, transform, names, device)


def predict_bytes(data, model_name, root=ROOT, device="cpu"):
    return predict_image(read_image(data), model_name, root, device)
