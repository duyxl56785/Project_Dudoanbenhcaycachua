"""Generate class distribution and training augmentation example from real NV1 data.

Run: python -m scripts.visualize_dataset --data-root data/raw
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from PIL import Image

from src.data.transforms import build_train_transform

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot"]


def run(data_root: Path, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with (ROOT / "data/dataset_manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = Counter((r["split"], r["label"]) for r in rows)
    folders = ["Tomato___healthy", "Tomato___Early_blight",
               "Tomato___Late_blight", "Tomato___Bacterial_spot"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(4)
    bottom = [0] * 4
    for split in ("train", "val", "test"):
        values = [counts[(split, label)] for label in folders]
        ax.bar(x, values, bottom=bottom, label=split)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_xticks(list(x), labels=NAMES)
    ax.set_ylabel("Images")
    ax.set_title("NV1 PlantVillage – 6502 selected images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=180)
    plt.close(fig)

    config = yaml.safe_load((ROOT / "configs/cnn.yaml").read_text(encoding="utf-8"))
    aug = build_train_transform("cnn", config, augment=True)
    sample = next(r for r in rows if r["split"] == "train")
    with Image.open(data_root / sample["path"]) as original:
        im = original.convert("RGB")
    # Show non-normalized augmentations to keep human-readable colors.
    from torchvision import transforms
    mean, std = config["preprocessing"]["normalize_mean"], config["preprocessing"]["normalize_std"]
    invert = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std],
    )
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    axes[0].imshow(im)
    axes[0].set_title("Original")
    for i in range(1, 5):
        tensor = invert(aug(im)).clamp(0, 1)
        axes[i].imshow(tensor.permute(1, 2, 0))
        axes[i].set_title(f"Train aug {i}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "augmentation_examples.png", dpi=150)
    plt.close(fig)
    print("Wrote class_distribution.png and augmentation_examples.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    run(args.data_root, args.figures_dir)
