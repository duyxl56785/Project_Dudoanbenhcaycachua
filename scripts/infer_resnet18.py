"""NV3 inference CLI and reusable predict function for NV4's Streamlit demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.resnet18_workflow import (REPO_ROOT, load_checkpoint, predict_image,
                                   read_config, read_labels)


def run(image: Path, config_path: Path, checkpoint_path: Path | None = None) -> dict:
    config = read_config(config_path)
    labels = read_labels(REPO_ROOT / config["data"]["class_mapping"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = checkpoint_path or REPO_ROOT / config["checkpoint"]["path"]
    model, _ = load_checkpoint(checkpoint, config, labels, device)
    return predict_image(model, image, config, labels, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="A JPG/PNG tomato leaf image")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/resnet18.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.image, args.config, args.checkpoint), indent=2, ensure_ascii=False))
