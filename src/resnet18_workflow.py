"""NV3-only dataset adapter and reusable inference/evaluation utilities.

Contract: NV1's CSV splits must exist with 'path' and 'label' columns; `path`
resolves relative to --data-root (default: repository root). No re-splitting.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset
from torchvision import transforms

from src.models.resnet18 import WEIGHTS_NAME, build_resnet18

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["model"]["num_classes"] != 4 or config["model"]["weights"] != WEIGHTS_NAME:
        raise ValueError("Unexpected classes/weights in NV3 configuration")
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


def image_transform(config: dict, *, train: bool) -> transforms.Compose:
    p = config["preprocessing"]
    ops: list = [transforms.Resize(p["resize_size"])]
    if train:
        ops.append(transforms.RandomCrop(p["image_size"]))
        if config["augmentation"]["train"].get("random_horizontal_flip"):
            ops.append(transforms.RandomHorizontalFlip())
        jitter = float(config["augmentation"]["train"].get("color_jitter", 0.0))
        if jitter:
            ops.append(transforms.ColorJitter(brightness=jitter, contrast=jitter))
    else:
        ops.append(transforms.CenterCrop(p["image_size"]))
    ops.extend((transforms.ToTensor(), transforms.Normalize(p["normalize_mean"], p["normalize_std"])))
    return transforms.Compose(ops)


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, data_root: Path, labels: list[str], transform: transforms.Compose):
        if not manifest.is_file():
            raise FileNotFoundError(f"NV1 split not available: {manifest}")
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"path", "label"}.issubset(reader.fieldnames):
                raise ValueError(f"Expected 'path' and 'label' CSV columns: {manifest}")
            records = list(reader)
        if not records:
            raise ValueError(f"Empty split manifest: {manifest}")
        self.samples: list[tuple[Path, int]] = []
        for row in records:
            raw_label = str(row["label"]).strip()
            if raw_label.isdigit() and int(raw_label) in range(len(labels)):
                label = int(raw_label)
            elif raw_label in labels:
                label = labels.index(raw_label)
            else:
                raise ValueError(f"Unknown label {raw_label!r} in {manifest}")
            raw_path = Path(row["path"].strip())
            resolved = raw_path if raw_path.is_absolute() else data_root / raw_path
            if not resolved.is_file():
                raise FileNotFoundError(f"Missing image listed in {manifest}: {resolved}")
            self.samples.append((resolved, label))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, label


def resolved_paths(config: dict, root: Path = REPO_ROOT) -> dict[str, Path]:
    return {key: root / config["data"][f"{key}_split"] for key in ("train", "val", "test")}


def checkpoint_payload(model, config: dict, labels: list[str], stage: str, best_val_f1: float) -> dict:
    return {"model_state_dict": model.state_dict(), "num_classes": 4,
            "labels": labels, "weights": config["model"]["weights"],
            "preprocessing": config["preprocessing"], "stage": stage,
            "best_val_macro_f1": float(best_val_f1)}


def load_checkpoint(path: Path, config: dict, labels: list[str], device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("labels") != labels or checkpoint.get("num_classes") != 4:
        raise ValueError("Checkpoint class mapping does not match shared config")
    if checkpoint.get("preprocessing") != config["preprocessing"] or checkpoint.get("weights") != WEIGHTS_NAME:
        raise ValueError("Checkpoint preprocessing/weights differs from NV3 config")
    model = build_resnet18(num_classes=4, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


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
