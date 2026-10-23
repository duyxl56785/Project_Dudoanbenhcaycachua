"""NV3 final test evaluation: run only after selecting a checkpoint on Validation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.resnet18_workflow import (REPO_ROOT, ManifestDataset, calculate_metrics,
                                   image_transform, load_checkpoint, read_config,
                                   read_labels, resolved_paths)


def run(config_path: Path, data_root: Path, checkpoint_path: Path | None = None) -> dict:
    config = read_config(config_path)
    labels = read_labels(REPO_ROOT / config["data"]["class_mapping"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = checkpoint_path or REPO_ROOT / config["checkpoint"]["path"]
    if not path.is_file():
        raise FileNotFoundError(f"No trained NV3 checkpoint found: {path}")
    model, metadata = load_checkpoint(path, config, labels, device)
    dataset = ManifestDataset(resolved_paths(config)["test"], data_root, labels, image_transform(config, train=False))
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                        num_workers=config["training"]["num_workers"])
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            y_true.extend(targets.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    metrics = calculate_metrics(y_true, y_pred, labels)
    metrics["stage"] = metadata["stage"]
    results_dir = REPO_ROOT / config["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "resnet18_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (results_dir / "resnet18_test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "true_label", "predicted_label"])
        for (image_path, _), actual, pred in zip(dataset.samples, y_true, y_pred):
            writer.writerow([str(image_path), actual, pred])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(metrics["confusion_matrix"], cmap="Blues")
    ax.set_xticks(range(4), labels=labels, rotation=30, ha="right")
    ax.set_yticks(range(4), labels=labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i, row in enumerate(metrics["confusion_matrix"]):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(results_dir / "resnet18_confusion_matrix.png", dpi=150)
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/resnet18.yaml")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    scores = run(args.config, args.data_root, args.checkpoint)
    print(json.dumps({"accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"]}, indent=2))
