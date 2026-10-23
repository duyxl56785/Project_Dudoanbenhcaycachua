"""Huấn luyện CNN từ đầu trên split Train/Validation của NV1.

Phụ trách: NV2 - Nguyễn Văn Duy.

Chạy từ thư mục gốc dự án:
    python -m scripts.train_cnn --config configs/cnn.yaml --data-root . --run-name baseline
Thử nghiệm có kiểm soát (mỗi lần đổi một yếu tố, chọn bằng Validation):
    python -m scripts.train_cnn ... --run-name e1_lr3e-4 --learning-rate 0.0003
    python -m scripts.train_cnn ... --run-name e2_no_aug --no-augmentation

Mỗi run ghi vào results/cnn/runs/<run-name>/: cnn_best.pt (+ .meta.json),
train_history.csv, run_info.json, cnn_train_curves.png.
Script KHÔNG đọc tập Test.
"""
import argparse
import copy
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.cnn_training import (EarlyStopping, load_class_names, run_epoch,  # noqa: E402
                              save_checkpoint, set_seed, use_utf8_output)
from src.models.cnn import build_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_FIELDS = ["epoch", "train_loss", "train_accuracy", "train_macro_f1",
                  "val_loss", "val_accuracy", "val_macro_f1", "lr"]


def git_commit():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment(device):
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "torchvision": torchvision.__version__, "device": str(device)}
    if str(device).startswith("cuda"):
        env["gpu"] = torch.cuda.get_device_name(torch.device(device))
    return env


def make_optimizer(model, training):
    if training["optimizer"] != "Adam":
        raise ValueError(f"Chỉ hỗ trợ optimizer Adam, cấu hình là {training['optimizer']}")
    return torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]),
                            weight_decay=float(training["weight_decay"]))


def train(model, train_loader, val_loader, config, out_dir, ckpt_path, device):
    """Huấn luyện, chọn checkpoint theo val_macro_f1, ghi lịch sử từng epoch."""
    if config["checkpoint"]["monitor"] != "val_macro_f1" or config["checkpoint"]["mode"] != "max":
        raise ValueError("Checkpoint phải chọn theo val_macro_f1 (mode max)")
    training = config["training"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(REPO_ROOT / config["data"]["class_mapping"])
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, training)
    early = EarlyStopping(int(training["early_stopping_patience"]), mode="max")
    best_epoch, epochs_run = None, 0
    commit, env = git_commit(), environment(device)

    with (out_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for epoch in range(1, int(training["epochs"]) + 1):
            tr = run_epoch(model, train_loader, criterion, device, optimizer)
            va = run_epoch(model, val_loader, criterion, device)
            epochs_run = epoch
            writer.writerow({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                             **{f"train_{k}": v for k, v in tr.items()},
                             **{f"val_{k}": v for k, v in va.items()}})
            stream.flush()
            print(f"epoch {epoch}: train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
                  f"val_acc={va['accuracy']:.4f} val_macro_f1={va['macro_f1']:.4f}", flush=True)
            if early.step(va["macro_f1"], tiebreak=va["loss"]):
                best_epoch = epoch
                save_checkpoint(model, ckpt_path, {
                    "labels": class_names, "preprocessing": config["preprocessing"],
                    "epoch": epoch, "val_macro_f1": va["macro_f1"], "seed": training["seed"],
                    "config": config, "git_commit": commit, "environment": env})
            if early.should_stop:
                print(f"Early stopping sau epoch {epoch}", flush=True)
                break
    return {"best_epoch": best_epoch, "best_val_macro_f1": early.best, "epochs_run": epochs_run}


def plot_curves(history_csv, out_png):
    with open(history_csv, encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "val"):
            ax.plot(epochs, [float(r[f"{split}_{metric}"]) for r in rows], marker="o",
                    label=f"{split} {metric}")
        ax.set_xlabel("epoch")
        ax.set_title(metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def load_nv1_data_api():
    try:
        from src.data.dataset import ManifestDataset
        from src.data.transforms import build_train_transform, build_val_transform
    except ImportError as exc:
        raise SystemExit(f"Chưa có module dữ liệu của NV1 (Gate 1): {exc}") from exc
    return ManifestDataset, build_train_transform, build_val_transform


def run(config_path, data_root, run_name, learning_rate=None, no_augmentation=False):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    run_config = copy.deepcopy(config)
    if learning_rate is not None:
        run_config["training"]["learning_rate"] = learning_rate
    training = run_config["training"]
    set_seed(int(training["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ManifestDataset, build_train_transform, build_val_transform = load_nv1_data_api()
    train_tf = build_train_transform("cnn", run_config, augment=not no_augmentation)
    val_tf = build_val_transform("cnn", run_config)
    data_root = Path(data_root)
    train_set = ManifestDataset(REPO_ROOT / run_config["data"]["train_split"], data_root, train_tf)
    val_set = ManifestDataset(REPO_ROOT / run_config["data"]["val_split"], data_root, val_tf)
    loader_args = {"batch_size": int(training["batch_size"]), "num_workers": int(training["num_workers"])}
    train_loader = DataLoader(train_set, shuffle=True,
                              generator=torch.Generator().manual_seed(int(training["seed"])), **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    out_dir = REPO_ROOT / run_config["output"]["results_dir"] / "runs" / run_name
    start = time.perf_counter()
    summary = train(build_model(4), train_loader, val_loader, run_config, out_dir,
                    out_dir / "cnn_best.pt", device)
    seconds = time.perf_counter() - start
    plot_curves(out_dir / "train_history.csv", out_dir / "cnn_train_curves.png")
    info = {"run_name": run_name, "summary": summary, "train_seconds": round(seconds, 1),
            "overrides": {"learning_rate": learning_rate, "no_augmentation": no_augmentation},
            "n_train": len(train_set), "n_val": len(val_set), "data_root": str(data_root),
            "git_commit": git_commit(), "environment": environment(device), "config": run_config}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def main(argv=None):
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/cnn.yaml")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--no-augmentation", action="store_true")
    args = parser.parse_args(argv)
    info = run(args.config, args.data_root, args.run_name, args.learning_rate, args.no_augmentation)
    print(json.dumps(info["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
