"""Đánh giá checkpoint CNN đã chốt trên Validation hoặc Test.

Phụ trách: NV2 - Nguyễn Văn Duy.

Chạy từ thư mục gốc dự án:
    python -m scripts.eval_cnn --config configs/cnn.yaml --data-root . --split test
Chỉ chạy --split test MỘT LẦN, sau khi đã chốt checkpoint bằng Validation.

Ghi vào results/cnn/ (split test): cnn_metrics.json, cnn_test_predictions.csv,
cnn_confusion_matrix.png. Split val dùng tiền tố cnn_val_.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.cnn_training import (compute_metrics, load_checkpoint,  # noqa: E402
                              load_class_names, use_utf8_output)

REPO_ROOT = Path(__file__).resolve().parents[1]


def evaluate(model, loader, class_names, device):
    """Dự đoán toàn bộ loader (không xáo trộn); trả về metrics và từng dòng dự đoán."""
    model = model.to(device).eval()
    y_true, y_pred, rows = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu()
            for label, p in zip(labels.tolist(), probs):
                pred = int(p.argmax())
                row = {"index": len(rows), "true": label, "pred": pred}
                row.update({f"p_{name}": float(v) for name, v in zip(class_names, p)})
                rows.append(row)
                y_true.append(label)
                y_pred.append(pred)
    if not rows:
        raise ValueError("Loader rỗng")
    return compute_metrics(y_true, y_pred, class_names), rows


def plot_confusion_matrix(matrix, class_names, out_png, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    peak = max(max(r) for r in matrix) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, value, ha="center", va="center",
                    color="white" if value > peak / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run(config_path, data_root, split, checkpoint=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    class_names = load_class_names(REPO_ROOT / config["data"]["class_mapping"])
    checkpoint = Path(checkpoint or REPO_ROOT / config["checkpoint"]["path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(checkpoint, class_names, device)
    if meta["preprocessing"] != config["preprocessing"]:
        raise SystemExit("preprocessing trong checkpoint khác configs/cnn.yaml; dùng đúng config của run đã chọn")

    try:
        from src.data.dataset import ManifestDataset
        from src.data.transforms import build_val_transform
    except ImportError as exc:
        raise SystemExit(f"Chưa có module dữ liệu của NV1 (Gate 1): {exc}") from exc
    dataset = ManifestDataset(REPO_ROOT / config["data"][f"{split}_split"], Path(data_root),
                              build_val_transform("cnn", config))
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False,
                        num_workers=int(config["training"]["num_workers"]))
    metrics, rows = evaluate(model, loader, class_names, device)
    for row in rows:
        row["path"] = str(dataset.samples[row["index"]][0])

    out_dir = REPO_ROOT / config["output"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "cnn" if split == "test" else f"cnn_{split}"
    report = {"split": split, "n_samples": len(rows), "checkpoint": str(checkpoint),
              "checkpoint_epoch": meta["epoch"], "checkpoint_val_macro_f1": meta["val_macro_f1"],
              "checkpoint_git_commit": meta["git_commit"], "device": device, **metrics}
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    fields = ["index", "path", "true", "pred"] + [f"p_{n}" for n in class_names]
    with (out_dir / f"cnn_{split}_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names,
                          out_dir / f"{prefix}_confusion_matrix.png", f"CNN - {split}")
    return report


def main(argv=None):
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/cnn.yaml")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    report = run(args.config, args.data_root, args.split, args.checkpoint)
    print(json.dumps({k: report[k] for k in ("split", "n_samples", "accuracy", "macro_f1")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
