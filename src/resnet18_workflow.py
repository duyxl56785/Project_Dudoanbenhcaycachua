"""NV3-only dataset adapter and reusable inference/evaluation utilities.

Contract: NV1's CSV splits must exist with 'path' and 'label' columns; `path`
resolves relative to --data-root (CLI default: data/raw). No re-splitting.
Dataset and transforms are delegated to NV1 PR #8.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.models.resnet18 import WEIGHTS_NAME, build_resnet18

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["model"]["num_classes"] != 4 or config["model"]["weights"] != WEIGHTS_NAME:
        raise ValueError("Unexpected classes/weights in NV3 configuration")
    if not config["model"].get("pretrained"):
        raise ValueError("Training requires ImageNet pretrained weights")
    if config["checkpoint"].get("mode") != "max":
        raise ValueError("Validation macro F1 must be maximized")
    if any(stage.get("lr_scheduler") is not None for stage in config["stages"].values()):
        raise ValueError("Learning rate schedulers are not implemented; use null")
    if config["checkpoint"]["monitor"] != "val_macro_f1":
        raise ValueError("Checkpoints must be selected on validation macro F1")
    return config


def read_labels(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as stream:
        mapping = json.load(stream)
    if sorted(mapping, key=int) != [str(i) for i in range(4)]:
        raise ValueError("Expected keys 0,1,2,3 in class_mapping.json")
    labels = [mapping[str(i)] for i in range(4)]
    if len(set(labels)) != 4:
        raise ValueError("Class labels are not unique")
    return labels


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shared_data_api():
    """Require NV1's reviewed implementation instead of duplicating its pipeline."""
    from src.data import dataset, transforms as shared_transforms
    required = (hasattr(dataset, "ManifestDataset"),
                hasattr(shared_transforms, "build_train_transform"),
                hasattr(shared_transforms, "build_val_transform"))
    if not all(required):
        raise RuntimeError("NV1 PR #8 shared Dataset/transforms must be integrated before running NV3 data workflows")
    return dataset, shared_transforms


def image_transform(config: dict, *, train: bool):
    _, shared = _shared_data_api()
    factory = shared.build_train_transform if train else shared.build_val_transform
    return factory("resnet18", config)


def ManifestDataset(manifest: Path, data_root: Path, labels: list[str], transform):
    """Compatibility wrapper for NV3 callers; all data validation belongs to NV1."""
    shared, _ = _shared_data_api()
    dataset = shared.ManifestDataset(manifest, data_root, transform=transform)
    if dataset.labels != labels:
        raise ValueError("NV1 class mapping differs from NV3 labels")
    return dataset


def resolved_paths(config: dict, root: Path = REPO_ROOT) -> dict[str, Path]:
    return {key: root / config["data"][f"{key}_split"] for key in ("train", "val", "test")}


def checkpoint_payload(model, config: dict, labels: list[str], stage: str, best_val_f1: float) -> dict:
    return {"model_state_dict": model.state_dict(), "num_classes": 4,
            "labels": labels, "weights": config["model"]["weights"],
            "preprocessing": config["preprocessing"], "stage": stage,
            "best_val_macro_f1": float(best_val_f1)}


def save_checkpoint(path: Path, model, metadata: dict) -> None:
    """Save the agreed demo format and its traceable metadata sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    path.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def load_checkpoint(path: Path, config: dict, labels: list[str], device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        # Legacy internal artifacts remain readable; never rewrite them implicitly.
        metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
        state = checkpoint["model_state_dict"]
    else:
        sidecar = path.with_suffix(".meta.json")
        if not sidecar.is_file():
            raise FileNotFoundError(f"Missing checkpoint metadata: {sidecar}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        state = checkpoint
    if metadata.get("labels") != labels or metadata.get("num_classes") != 4:
        raise ValueError("Checkpoint class mapping does not match shared config")
    if metadata.get("preprocessing") != config["preprocessing"] or metadata.get("weights") != WEIGHTS_NAME:
        raise ValueError("Checkpoint preprocessing/weights differs from NV3 config")
    model = build_resnet18(num_classes=4, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, metadata


def calculate_metrics(y_true: list[int], y_pred: list[int], labels: list[str]) -> dict:
    if not y_true:
        raise ValueError("No predictions to evaluate")
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(4)), average="macro", zero_division=0)),
            "classification_report": classification_report(y_true, y_pred, labels=list(range(4)),
                                                         target_names=labels, zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist()}


def predict_image(model, path: Path, config: dict, labels: list[str], device: torch.device) -> dict:
    with Image.open(path) as image:
        x = image_transform(config, train=False)(image.convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        scores = torch.softmax(logits, dim=1)[0].cpu().tolist()
    return {"label": labels[int(np.argmax(scores))], "scores": {name: float(score) for name, score in zip(labels, scores)}}
