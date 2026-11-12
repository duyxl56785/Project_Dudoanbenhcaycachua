"""HOG + linear SVM baseline on the fixed NV1 tomato-leaf splits."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog


@dataclass(frozen=True)
class HOGConfig:
    image_size: int = 64
    grayscale: bool = True
    orientations: int = 9
    pixels_per_cell: int = 8
    cells_per_block: int = 2
    block_norm: str = "L2-Hys"

    def dimension(self) -> int:
        cells = self.image_size // self.pixels_per_cell
        blocks = cells - self.cells_per_block + 1
        if self.image_size < 16 or self.image_size % self.pixels_per_cell or blocks < 1:
            raise ValueError("Invalid HOG image/cell/block size")
        if self.orientations < 2 or self.block_norm not in ("L1", "L1-sqrt", "L2", "L2-Hys"):
            raise ValueError("Invalid HOG orientations or block norm")
        # scikit-image takes the strongest gradient over RGB channels; it does
        # not concatenate three separate descriptor vectors.
        return blocks * blocks * self.cells_per_block**2 * self.orientations


def extract_hog(image: Image.Image, config: HOGConfig) -> np.ndarray:
    """Use a deterministic HOG pipeline, independent of CNN normalization."""
    dimension = config.dimension()
    if not isinstance(image, Image.Image):
        raise ValueError("Expected a PIL image")
    mode = "L" if config.grayscale else "RGB"
    pixels = np.asarray(image.convert(mode).resize((config.image_size,) * 2, Image.Resampling.BILINEAR))
    kwargs = {"orientations": config.orientations,
              "pixels_per_cell": (config.pixels_per_cell,) * 2,
              "cells_per_block": (config.cells_per_block,) * 2,
              "block_norm": config.block_norm, "feature_vector": True}
    vector = hog(pixels, **kwargs, channel_axis=None if config.grayscale else -1).astype(np.float32)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError("Invalid HOG vector shape or non-finite values")
    return vector


def image_hog(path: Path, config: HOGConfig) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return extract_hog(image, config)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"Cannot read image {path}: {exc}") from exc


def load_mapping(project_root: Path) -> tuple[str, ...]:
    mapping = json.loads((project_root / "configs/class_mapping.json").read_text(encoding="utf-8"))
    if set(mapping) != {"0", "1", "2", "3"}:
        raise ValueError("Expected the four official class indices")
    names = tuple(mapping[str(i)] for i in range(4))
    if len(set(names)) != 4:
        raise ValueError("Class names must be unique")
    return names


def read_splits(project_root: Path, data_root: Path, *, check_files: bool = True) -> dict[str, list[dict[str, str]]]:
    """Read the existing split manifests and reject path/group/hash leakage."""
    splits = {}
    paths: set[str] = set()
    groups: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        with (project_root / "data/splits" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty {split} split")
        for row in rows:
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts or row["path"] in paths:
                raise ValueError(f"Unsafe or duplicated path: {row['path']}")
            if row["label"] not in {"0", "1", "2", "3"}:
                raise ValueError(f"Invalid class index: {row['label']}")
            paths.add(row["path"])
            if check_files and not (data_root / relative).is_file():
                raise FileNotFoundError(data_root / relative)
        splits[split] = rows
        groups[split] = {row["group"] for row in rows}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[left] & groups[right]:
            raise ValueError(f"Group leakage between {left} and {right}")
    manifest_path = project_root / "data/dataset_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = {row["path"]: row for row in csv.DictReader(handle)}
    hashes = {}
    folder_by_index = {"0": "Tomato___healthy", "1": "Tomato___Early_blight",
                       "2": "Tomato___Late_blight", "3": "Tomato___Bacterial_spot"}
    for split, rows in splits.items():
        if any(row["path"] not in manifest or manifest[row["path"]]["split"] != split
               or manifest[row["path"]]["label"] != folder_by_index[row["label"]]
               or manifest[row["path"]]["group_id"] != row["group"] for row in rows):
            raise ValueError(f"Manifest mismatch for {split}")
        hashes[split] = {manifest[row["path"]]["sha256"] for row in rows}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if hashes[left] & hashes[right]:
            raise ValueError(f"SHA-256 leakage between {left} and {right}")
    return splits


def feature_matrix(rows: list[dict[str, str]], data_root: Path, config: HOGConfig) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.empty((len(rows), config.dimension()), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        matrix[index] = image_hog(data_root / row["path"], config)
        labels[index] = int(row["label"])
    return matrix, labels


def config_dict(config: HOGConfig) -> dict:
    return asdict(config)
