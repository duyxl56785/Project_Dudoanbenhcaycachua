"""Shared image transforms for CNN and ResNet18 (NV1).

Validation, Test and Streamlit MUST use build_val_transform(model_name, config).
The CNN baseline matches the validated Colab 224x224 normalization pipeline.
ResNet18 matches NV3 PR #7: resize shorter side to 256, center-crop 224.
"""
from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_MODELS = {"cnn", "resnet18"}


def _preprocessing(model_name: str, config: dict) -> tuple[int, int, list[float], list[float]]:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name!r}")
    p = config.get("preprocessing", {})
    size = int(p.get("image_size") or 224)
    resize_size = int(p.get("resize_size") or 256)
    mean = list(p.get("normalize_mean") or IMAGENET_MEAN)
    std = list(p.get("normalize_std") or IMAGENET_STD)
    if size <= 0 or resize_size <= 0 or len(mean) != 3 or len(std) != 3:
        raise ValueError("Invalid image_size, resize_size or normalization channels")
    if any(s <= 0 for s in std):
        raise ValueError("Normalization standard deviations must be positive")
    return size, resize_size, mean, std


def build_val_transform(model_name: str, config: dict):
    """Deterministic PIL RGB -> CHW tensor for Validation/Test/Inference."""
    size, resize_size, mean, std = _preprocessing(model_name, config)
    if model_name == "resnet18":
        ops = [transforms.Resize(resize_size), transforms.CenterCrop(size)]
    else:
        ops = [transforms.Resize((size, size))]
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(ops)


def build_train_transform(model_name: str, config: dict, augment: bool = True):
    """Train transform. Random augmentation never applies to Val/Test."""
    size, resize_size, mean, std = _preprocessing(model_name, config)
    if model_name == "resnet18":
        ops = [transforms.Resize(resize_size)]
        if augment:
            ops += [
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
            ]
        else:
            ops += [transforms.CenterCrop(size)]
    else:
        ops = [transforms.Resize((size, size))]
        if augment:
            ops += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15, fill=(128, 128, 128)),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08),
            ]
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(ops)
