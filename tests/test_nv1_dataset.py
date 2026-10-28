"""NV1 data contract smoke tests (no 100 MB image archive required).

Run: python -m pytest -q tests/test_nv1_dataset.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import yaml
from PIL import Image

from src.data.dataset import ManifestDataset, create_dataloaders, read_class_mapping
from src.data.transforms import build_val_transform
from scripts.check_dataset import check, ROOT


def test_manifest_integrity():
    summary = check()
    assert summary["split_counts"] == {"train": 4115, "val": 1031, "test": 1356}
    assert summary["unique_sha256"] == 6502


def test_mapping_and_val_transform():
    assert read_class_mapping() == [
        "Healthy", "Early Blight", "Late Blight", "Bacterial Spot"
    ]
    config = yaml.safe_load((ROOT / "configs/cnn.yaml").read_text(encoding="utf-8"))
    val_transform = build_val_transform("cnn", config)
    im = Image.new("RGB", (256, 256), (66, 120, 58))
    x1, x2 = val_transform(im), val_transform(im)
    assert x1.shape == (3, 224, 224)
    assert torch.equal(x1, x2)


def test_loaders_with_synthetic_files(tmp_path: Path):
    data_root = tmp_path / "images"
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    rel = "raw/color/Tomato___healthy/test_photo.JPG"
    full = data_root / rel
    full.parent.mkdir(parents=True)
    Image.new("RGB", (256, 256), (12, 80, 112)).save(full, format="JPEG")
    for split in ("train", "val", "test"):
        with (split_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(("path", "label", "group"))
            writer.writerow((rel, 0, "group0"))
    for model in ("cnn", "resnet18"):
        loaders, datasets, mapping = create_dataloaders(
            data_root, model_name=model, manifest_dir=split_dir,
            batch_size=1, num_workers=0, augment=False
        )
        assert len(datasets["train"]) == len(datasets["val"]) == len(datasets["test"]) == 1
        x, y = next(iter(loaders["val"]))
        assert tuple(x.shape) == (1, 3, 224, 224)
        assert y.tolist() == [0]
        assert mapping["Healthy"] == 0 and mapping["Bacterial Spot"] == 3
