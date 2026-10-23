"""Shared PlantVillage manifest Dataset/DataLoader (NV1).

CSV schema: path,label[,group]; paths are relative to data_root (ZIP extraction root).
Classes use configs/class_mapping.json, NEVER alphabetical ImageFolder ordering.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile, BadZipFile, PurePosixPath

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import build_train_transform, build_val_transform

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASS_FOLDERS = {
    "Tomato___healthy": 0,
    "Tomato___Early_blight": 1,
    "Tomato___Late_blight": 2,
    "Tomato___Bacterial_spot": 3,
}
EXPECTED_LABELS = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot"]


def read_class_mapping(path: str | Path = REPO_ROOT / "configs/class_mapping.json") -> list[str]:
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(mapping) != {str(i) for i in range(4)}:
        raise ValueError("class_mapping.json must define keys 0,1,2,3")
    labels = [mapping[str(i)] for i in range(4)]
    if labels != EXPECTED_LABELS:
        raise ValueError(f"Unexpected label mapping: {labels}")
    return labels


def prepare_archive(archive_path: str | Path, data_root: str | Path) -> Path:
    """Extract trusted NV1 archive into data_root, verifying listed members exist.

    Do not read ZIP archives from untrusted sources. This function rejects traversal
    members and checks representative files but is not a decompression-bomb defense.
    """
    archive_path, data_root = Path(archive_path), Path(data_root)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    data_root.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(archive_path) as z:
            for member in z.namelist():
                p = PurePosixPath(member)
                if p.is_absolute() or ".." in p.parts or not member:
                    raise ValueError(f"Unsafe ZIP member: {member}")
            z.extractall(data_root)
    except BadZipFile as exc:
        raise ValueError(f"Invalid dataset archive: {archive_path}") from exc
    return data_root


class ManifestDataset(Dataset):
    """Reads original 256x256 JPGs using the immutable CSV split."""
    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        transform=None,
        mapping_path: str | Path = REPO_ROOT / "configs/class_mapping.json",
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.labels = read_class_mapping(mapping_path)
        self.samples: list[tuple[Path, int]] = []
        with Path(manifest).open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"path", "label"}.issubset(reader.fieldnames):
                raise ValueError(f"CSV needs columns path,label: {manifest}")
            for row in reader:
                raw_path = row["path"].strip()
                raw_label = row["label"].strip()
                if raw_label in {"0", "1", "2", "3"}:
                    label = int(raw_label)
                elif raw_label in self.labels:
                    label = self.labels.index(raw_label)
                else:
                    raise ValueError(f"Unsupported label {raw_label!r} in {manifest}")
                rel_path = PurePosixPath(raw_path)
                if rel_path.is_absolute() or ".." in rel_path.parts:
                    raise ValueError(f"Unsafe image path: {raw_path}")
                if CLASS_FOLDERS.get(rel_path.parent.name) != label:
                    raise ValueError(f"Label/path mismatch: {raw_path} / {label}")
                path = self.data_root.joinpath(*rel_path.parts)
                if not path.is_file():
                    raise FileNotFoundError(f"Image missing from dataset: {path}")
                self.samples.append((path, label))
        if not self.samples:
            raise ValueError(f"Empty split manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def create_dataloaders(
    data_root: str | Path,
    model_name: str = "cnn",
    config: dict | None = None,
    *,
    archive_path: str | Path | None = None,
    manifest_dir: str | Path = REPO_ROOT / "data/splits",
    batch_size: int = 32,
    augment: bool = True,
    num_workers: int = 0,
    seed: int = 42,
):
    """Return (loaders, datasets, class_to_idx) for BOTH training models.

    Extract official NV1 ZIP only if archive_path is explicitly supplied.
    The dataset root must contain raw/color/<Tomato class>/<image>.JPG.
    """
    if model_name not in {"cnn", "resnet18"}:
        raise ValueError(f"Unknown model_name: {model_name}")
    if config is None:
        with (REPO_ROOT / f"configs/{model_name}.yaml").open(encoding="utf-8") as fp:
            config = yaml.safe_load(fp)
    data_root = Path(data_root)
    if archive_path is not None:
        # Avoid overwriting a fully extracted dataset on repeated calls.
        if not (data_root / "raw" / "color").is_dir():
            prepare_archive(archive_path, data_root)
    manifest_dir = Path(manifest_dir)
    datasets = {}
    for split in ("train", "val", "test"):
        transform = (build_train_transform(model_name, config, augment)
                     if split == "train" else build_val_transform(model_name, config))
        datasets[split] = ManifestDataset(
            manifest_dir / f"{split}.csv", data_root, transform
        )
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"), num_workers=num_workers,
            generator=generator if split == "train" else None,
        )
        for split, ds in datasets.items()
    }
    return loaders, datasets, {name: i for i, name in enumerate(EXPECTED_LABELS)}
