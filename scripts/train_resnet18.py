"""NV3: Train ImageNet-pretrained ResNet18 in two stages on NV1 manifests.

From repository root: python -m scripts.train_resnet18 --data-root data/raw
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.models.resnet18 import build_resnet18, configure_stage, parameter_counts, training_mode
from src.resnet18_workflow import (REPO_ROOT, ManifestDataset, checkpoint_payload,
                                   image_transform, read_config, read_labels,
                                   resolved_paths, set_seed, load_checkpoint, save_checkpoint)


def epoch_pass(model, loader, criterion, device, optimizer=None) -> dict[str, float]:
    if optimizer is None:
        model.eval()
    else:
        training_mode(model)
    total_loss, seen, correct = 0.0, 0, 0
    true, predicted = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        batch_predictions = logits.argmax(1)
        n = labels.numel()
        total_loss += loss.item() * n
        seen += n
        correct += (batch_predictions == labels).sum().item()
        true.extend(labels.detach().cpu().tolist())
        predicted.extend(batch_predictions.detach().cpu().tolist())
    if not seen:
        raise ValueError("Empty dataset")
    return {"loss": total_loss / seen, "accuracy": correct / seen,
            "macro_f1": float(f1_score(true, predicted, labels=[0, 1, 2, 3],
                                        average="macro", zero_division=0))}


def optimizer_for_stage(model, config: dict, stage: str):
    cfg = config["stages"][stage]
    if cfg["optimizer"] != "AdamW":
        raise ValueError("Only AdamW is supported by the NV3 training script")
    if stage == "feature_extraction":
        return torch.optim.AdamW(model.fc.parameters(), lr=float(cfg["learning_rate"]),
                                 weight_decay=float(cfg["weight_decay"]))
    backbone = [p for name, p in model.named_parameters() if not name.startswith("fc.") and p.requires_grad]
    return torch.optim.AdamW([
        {"params": backbone, "lr": float(cfg["lr_backbone"])},
        {"params": list(model.fc.parameters()), "lr": float(cfg["lr_head"])}
    ], weight_decay=float(cfg["weight_decay"]))


def plot_curves(records: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax in zip(("loss", "accuracy"), axes):
        for stage in ("feature_extraction", "fine_tuning"):
            sub = [r for r in records if r["stage"] == stage]
            for split in ("train", "val"):
                ax.plot([r["epoch"] for r in sub], [r[f"{split}_{metric}"] for r in sub],
                        label=f"{stage}/{split}")
        ax.set_xlabel("Epoch within stage")
        ax.set_ylabel(metric.title())
        ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(config_path: Path, data_root: Path) -> dict:
    config = read_config(config_path)
    delivery = REPO_ROOT / config["checkpoint"]["path"]
    result_root = REPO_ROOT / config["output"]["results_dir"]
    protected = [delivery, delivery.with_suffix(".meta.json")]
    for stage in ("feature_extraction", "fine_tuning"):
        protected.extend([result_root / f"{stage}_best.pt", result_root / f"{stage}_best.meta.json"])
        if int(config["stages"][stage]["epochs"]) < 1:
            raise ValueError("Each training stage requires at least one epoch")
    if any(path.exists() for path in protected):
        raise FileExistsError("Existing checkpoints are preserved; configure new checkpoint and results paths for a new run")
    labels = read_labels(REPO_ROOT / config["data"]["class_mapping"])
    set_seed(int(config["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = resolved_paths(config)
    for name in ("train", "val"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"Wait for NV1 Gate 1: missing {paths[name]}")
    datasets = {
        name: ManifestDataset(paths[name], data_root, labels, image_transform(config, train=name == "train"))
        for name in ("train", "val")
    }
    loaders = {name: DataLoader(ds, batch_size=config["training"]["batch_size"],
                                shuffle=(name == "train"), num_workers=config["training"]["num_workers"])
               for name, ds in datasets.items()}
    model = build_resnet18(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    results_dir = REPO_ROOT / config["output"]["results_dir"]
    checkpoint_path = REPO_ROOT / config["checkpoint"]["path"]
    figures_dir = REPO_ROOT / config["output"]["figures_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    records, comparison = [], []
    stage_checkpoints = {}

    for stage in ("feature_extraction", "fine_tuning"):
        if stage == "fine_tuning":
            # Start from the strongest feature-extraction checkpoint, not its last epoch.
            model, _ = load_checkpoint(stage_checkpoints["feature_extraction"], config, labels, device)
        stage_cfg = config["stages"][stage]
        configure_stage(model, stage, tuple(config["stages"]["fine_tuning"]["unfreeze_layers"]))
        counts = parameter_counts(model)
        optimizer = optimizer_for_stage(model, config, stage)
        best_score, best_loss, stale = -1.0, float("inf"), 0
        stage_checkpoint = results_dir / f"{stage}_best.pt"
        for epoch in range(1, int(stage_cfg["epochs"]) + 1):
            train = epoch_pass(model, loaders["train"], criterion, device, optimizer=optimizer)
            val = epoch_pass(model, loaders["val"], criterion, device)
            record = {"stage": stage, "epoch": epoch,
                      **{f"train_{k}": v for k, v in train.items()},
                      **{f"val_{k}": v for k, v in val.items()}}
            records.append(record)
            print(f"{stage} epoch={epoch} train_loss={train['loss']:.4f} "
                  f"val_loss={val['loss']:.4f} val_macro_f1={val['macro_f1']:.4f}")
            improved = (val["macro_f1"] > best_score or
                        (val["macro_f1"] == best_score and val["loss"] < best_loss))
            if improved:
                best_score, best_loss, stale = val["macro_f1"], val["loss"], 0
                metadata = checkpoint_payload(model, config, labels, stage, best_score)
                metadata.pop("model_state_dict")
                metadata.update({"epoch": epoch, "val_loss": best_loss,
                                 "seed": int(config["training"]["seed"]), "config": config,
                                 "environment": {"python": platform.python_version(),
                                                 "torch": str(torch.__version__),
                                                 "torchvision": str(__import__("torchvision").__version__),
                                                 "device": str(device)}})
                save_checkpoint(stage_checkpoint, model, metadata)
            else:
                stale += 1
            patience = stage_cfg.get("early_stopping_patience")
            if patience is not None and stale >= int(patience):
                break
        comparison.append({"stage": stage, "val_macro_f1": best_score, "val_loss": best_loss,
                           "total_parameters": counts["total"], "trainable_parameters": counts["trainable"]})
        stage_checkpoints[stage] = stage_checkpoint

    # Select by Validation only; tie breaks on lower val loss.
    winner = sorted(comparison, key=lambda r: (-r["val_macro_f1"], r["val_loss"]))[0]
    selected_model, selected_metadata = load_checkpoint(stage_checkpoints[winner["stage"]], config, labels, device)
    if checkpoint_path.exists() or checkpoint_path.with_suffix(".meta.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing delivery checkpoint: {checkpoint_path}; configure a new path")
    save_checkpoint(checkpoint_path, selected_model, selected_metadata)
    with (results_dir / "resnet18_feature_vs_finetune.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    with (results_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    plot_curves(records, figures_dir / "resnet18_train_curves.png")
    details = {"selected_stage": winner["stage"], "validation_macro_f1": winner["val_macro_f1"],
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "torchvision": __import__("torchvision").__version__, "device": str(device)},
               "config": str(config_path)}
    (results_dir / "selected_stage.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NV3 ResNet18 on NV1 splits")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/resnet18.yaml")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.data_root), indent=2))
