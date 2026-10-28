"""Validate and compare CNN and ResNet18 predictions on one locked split.

The model owners produce prediction CSVs with ``scripts.eval_cnn`` and
``scripts.eval_resnet18``. This script validates that both files describe the
same samples and labels, then writes a traceable comparison. It never trains
or selects a checkpoint, so Test remains an evaluation-only split.

Example::

    python -m scripts.evaluate_models \
      --cnn-predictions results/cnn/cnn_test_predictions.csv \
      --resnet18-predictions results/resnet18/resnet18_test_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
except ImportError:  # pragma: no cover - exercised only in minimal environments
    accuracy_score = confusion_matrix = precision_recall_fscore_support = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = REPO_ROOT / "configs" / "class_mapping.json"
ID_COLUMNS = ("path", "image_path", "sample_id", "image_id", "id", "filename")
TRUE_COLUMNS = ("true", "true_label", "label", "target", "y_true")
PRED_COLUMNS = ("pred", "predicted", "predicted_label", "prediction", "y_pred")


def _first_column(fieldnames: Iterable[str], candidates: Iterable[str], role: str) -> str:
    fields = {name.strip().lower(): name for name in fieldnames if name}
    for candidate in candidates:
        if candidate in fields:
            return fields[candidate]
    raise ValueError(f"Không tìm thấy cột {role}; cho phép: {', '.join(candidates)}")


def load_class_mapping(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"class_mapping phải là object không rỗng: {path}")
    mapping: dict[int, str] = {}
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Nhãn không phải số nguyên: {key!r}") from exc
        if not isinstance(value, str) or not value.strip() or index in mapping:
            raise ValueError(f"Mapping không hợp lệ tại nhãn {key!r}")
        mapping[index] = value.strip()
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError(f"class_mapping phải liên tục từ 0; nhận {sorted(mapping)}")
    return dict(sorted(mapping.items()))


def normalize_label(value: Any, mapping: dict[int, str], source: Path) -> int:
    text = str(value).strip()
    try:
        index = int(text)
    except ValueError:
        names = {name.casefold(): idx for idx, name in mapping.items()}
        if text.casefold() not in names:
            raise ValueError(f"Nhãn không thuộc mapping trong {source}: {value!r}")
        index = names[text.casefold()]
    if index not in mapping:
        raise ValueError(f"Nhãn ngoài mapping trong {source}: {index}")
    return index


def load_predictions(path: Path, mapping: dict[int, str]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy prediction CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV không có header: {path}")
        true_column = _first_column(reader.fieldnames, TRUE_COLUMNS, "ground truth")
        pred_column = _first_column(reader.fieldnames, PRED_COLUMNS, "prediction")
        lookup = {name.strip().lower(): name for name in reader.fieldnames if name}
        id_column = next((lookup[name] for name in ID_COLUMNS if name in lookup), None)
        rows, seen = [], set()
        for row_number, row in enumerate(reader, start=2):
            sample_id = str(row[id_column]).strip() if id_column and row.get(id_column) is not None else str(row_number - 2)
            if not sample_id:
                raise ValueError(f"Mẫu rỗng ở dòng {row_number} trong {path}")
            if sample_id in seen:
                raise ValueError(f"Mẫu trùng {sample_id!r} ở dòng {row_number} trong {path}")
            seen.add(sample_id)
            rows.append({"id": sample_id,
                         "true": normalize_label(row[true_column], mapping, path),
                         "pred": normalize_label(row[pred_column], mapping, path)})
    if not rows:
        raise ValueError(f"Prediction CSV rỗng: {path}")
    return rows


def load_metadata(predictions_path: Path, explicit_path: Path | None) -> dict[str, Any]:
    candidate = explicit_path
    name = predictions_path.name
    if candidate is None and name.endswith("_test_predictions.csv"):
        candidate = predictions_path.with_name(name.replace("_test_predictions.csv", "_metrics.json"))
    if candidate is None and name.endswith("_predictions.csv"):
        candidate = predictions_path.with_name(name.replace("_predictions.csv", "_metrics.json"))
    if candidate is None or not candidate.is_file():
        return {}
    data = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metadata phải là object: {candidate}")
    data["_metadata_file"] = str(candidate)
    return data


def evaluate(rows: list[dict[str, Any]], mapping: dict[int, str]) -> dict[str, Any]:
    if precision_recall_fscore_support is None:
        raise RuntimeError("Thiếu scikit-learn; chạy `pip install -r requirements.txt` trước khi đánh giá")
    labels = list(mapping)
    y_true, y_pred = [row["true"] for row in rows], [row["pred"] for row in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n_samples": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {mapping[index]: {"precision": float(precision[pos]),
                                        "recall": float(recall[pos]),
                                        "f1": float(f1[pos]),
                                        "support": int(support[pos])}
                      for pos, index in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def validate_same_test_set(cnn: list[dict[str, Any]], resnet: list[dict[str, Any]]) -> None:
    cnn_ids, resnet_ids = {row["id"] for row in cnn}, {row["id"] for row in resnet}
    if cnn_ids != resnet_ids:
        raise ValueError("Hai prediction CSV không dùng cùng Test set; "
                         f"thiếu ở CNN={sorted(resnet_ids - cnn_ids)[:5]}, "
                         f"thiếu ở ResNet18={sorted(cnn_ids - resnet_ids)[:5]}")
    truth = {row["id"]: row["true"] for row in cnn}
    mismatch = [row["id"] for row in resnet if truth[row["id"]] != row["true"]]
    if mismatch:
        raise ValueError(f"Ground truth khác nhau giữa hai CSV (ví dụ: {mismatch[:5]})")


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return ""


def plot_confusion(metrics: dict[str, Any], mapping: dict[int, str], path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Thiếu matplotlib; chạy `pip install -r requirements.txt` trước khi đánh giá") from exc
    matrix, names = metrics["confusion_matrix"], list(mapping.values())
    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(title)
    peak = max((max(row) for row in matrix), default=0) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > peak / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def comparison_row(name: str, split: str, metrics: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"model": name, "split": split, "n_samples": metrics["n_samples"],
        "accuracy": metrics["accuracy"], "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"], "macro_f1": metrics["macro_f1"],
        "total_parameters": _metadata_value(metadata, "total_parameters", "num_parameters", "parameters"),
        "trainable_parameters": _metadata_value(metadata, "trainable_parameters"),
        "train_seconds": _metadata_value(metadata, "train_seconds", "training_seconds"),
        "inference_seconds": _metadata_value(metadata, "inference_seconds", "test_seconds"),
        "device": _metadata_value(metadata, "device"), "checkpoint": _metadata_value(metadata, "checkpoint"),
        "metadata_file": _metadata_value(metadata, "_metadata_file")}
    for class_name, values in metrics["per_class"].items():
        row[f"{class_name}_precision"], row[f"{class_name}_recall"], row[f"{class_name}_f1"] = values["precision"], values["recall"], values["f1"]
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_analysis(path: Path, mapping: dict[int, str], split: str, metrics_by_model: dict[str, dict[str, Any]], metadata_by_model: dict[str, dict[str, Any]], files: dict[str, Path]) -> None:
    best = max(metrics_by_model, key=lambda name: metrics_by_model[name]["macro_f1"])
    pair_counts: Counter[tuple[str, str]] = Counter()
    for metrics in metrics_by_model.values():
        for i, row in enumerate(metrics["confusion_matrix"]):
            for j, value in enumerate(row):
                if i != j: pair_counts[(mapping[i], mapping[j])] += int(value)
    lines = ["# Phân tích đánh giá NV5", "", f"- Split: `{split}` (chỉ đánh giá sau khi chọn checkpoint bằng Validation).", f"- Tạo lúc: `{datetime.now(timezone.utc).isoformat()}`.", f"- Python: `{platform.python_version()}`.", f"- Mapping: `{', '.join(mapping.values())}`.", "", "## Nguồn truy vết", ""]
    for model, source in files.items():
        lines.append(f"- **{model}** prediction: `{source}`")
        if metadata_by_model[model].get("_metadata_file"): lines.append(f"  - metadata: `{metadata_by_model[model]['_metadata_file']}`")
        if metadata_by_model[model].get("checkpoint"): lines.append(f"  - checkpoint: `{metadata_by_model[model]['checkpoint']}`")
    lines += ["", "## Kết quả", "", "| Mô hình | Số mẫu | Accuracy | Macro Precision | Macro Recall | Macro F1 |", "|---|---:|---:|---:|---:|---:|"]
    for model, metrics in metrics_by_model.items():
        lines.append(f"| {model} | {metrics['n_samples']} | {metrics['accuracy']:.4f} | {metrics['macro_precision']:.4f} | {metrics['macro_recall']:.4f} | {metrics['macro_f1']:.4f} |")
    lines += ["", f"Macro F1 cao nhất trong file này: **{best}** ({metrics_by_model[best]['macro_f1']:.4f}). Đây không phải quyết định chọn lại checkpoint.", "", "## Cặp lớp dễ nhầm", ""]
    confusion_lines = [f"- {actual} → {predicted}: {count} mẫu" for (actual, predicted), count in pair_counts.most_common() if count]
    lines.extend(confusion_lines[:10] or ["- Không có mẫu nhầm lớp."])
    lines += ["", "## Hạn chế", "", "- PlantVillage là dữ liệu được kiểm soát; kết quả không đại diện trực tiếp cho ảnh ngoài thực địa.", "- Ảnh ngoài các lớp mapping vẫn có thể bị ép vào một trong bốn lớp.", "- Không có thí nghiệm bệnh mới chớm, vì vậy không kết luận khả năng phát hiện sớm.", "- Metadata còn trống nghĩa là chủ model chưa bàn giao số liệu; không tự điền số liệu minh họa.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    mapping = load_class_mapping(Path(args.class_mapping))
    cnn_path, resnet_path = Path(args.cnn_predictions), Path(args.resnet18_predictions)
    cnn_rows, resnet_rows = load_predictions(cnn_path, mapping), load_predictions(resnet_path, mapping)
    validate_same_test_set(cnn_rows, resnet_rows)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    metrics = {"CNN": evaluate(cnn_rows, mapping), "ResNet18": evaluate(resnet_rows, mapping)}
    metadata = {"CNN": load_metadata(cnn_path, Path(args.cnn_metrics) if args.cnn_metrics else None),
                "ResNet18": load_metadata(resnet_path, Path(args.resnet18_metrics) if args.resnet18_metrics else None)}
    for name, score in metrics.items():
        if metadata[name].get("split") and str(metadata[name]["split"]).lower() != args.split.lower():
            raise ValueError(f"Metadata của {name} ghi split={metadata[name]['split']!r}, không khớp --split {args.split!r}")
        if metadata[name].get("n_samples") is not None and int(metadata[name]["n_samples"]) != score["n_samples"]:
            raise ValueError(f"Số mẫu metadata của {name} không khớp prediction CSV")
    write_csv(out / "comparison.csv", [comparison_row(name, args.split, score, metadata[name]) for name, score in metrics.items()])
    plot_confusion(metrics["CNN"], mapping, out / "cnn_confusion_matrix.png", f"CNN - {args.split}")
    plot_confusion(metrics["ResNet18"], mapping, out / "resnet18_confusion_matrix.png", f"ResNet18 - {args.split}")
    write_analysis(out / "analysis.md", mapping, args.split, metrics, metadata, {"CNN": cnn_path, "ResNet18": resnet_path})
    return {"output_dir": str(out), "models": {name: {key: value for key, value in score.items() if key != "confusion_matrix"} for name, score in metrics.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cnn-predictions", required=True)
    parser.add_argument("--resnet18-predictions", required=True)
    parser.add_argument("--cnn-metrics")
    parser.add_argument("--resnet18-metrics")
    parser.add_argument("--class-mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
