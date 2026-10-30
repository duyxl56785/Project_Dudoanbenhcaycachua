"""Đánh giá độc lập và so sánh CNN/ResNet18 trên Test chung.

NV5 chỉ tổng hợp artifact đã bàn giao. Script không huấn luyện, không chọn
checkpoint và không tự điền số liệu còn thiếu. Mỗi model có thể được cung cấp
prediction CSV (tính lại metrics) hoặc metrics JSON có nguồn truy vết; nếu
không có artifact thì model được ghi ``pending`` trong bảng.
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
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
except ImportError:  # pragma: no cover
    accuracy_score = confusion_matrix = precision_recall_fscore_support = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = REPO_ROOT / "configs" / "class_mapping.json"
ID_COLUMNS = ("path", "image_path", "sample_id", "image_id", "id", "filename")
TRUE_COLUMNS = ("true", "true_label", "label", "target", "y_true")
PRED_COLUMNS = ("pred", "predicted", "predicted_label", "prediction", "y_pred")
SPLIT_COLUMNS = ("split", "dataset_split", "subset")
SCORE_KEYS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def _first_column(fieldnames: Iterable[str], candidates: Iterable[str], role: str) -> str:
    fields = {name.strip().lower(): name for name in fieldnames if name}
    for candidate in candidates:
        if candidate in fields:
            return fields[candidate]
    raise ValueError(f"Không tìm thấy cột {role}; cho phép: {', '.join(candidates)}")


def load_class_mapping(path: Path) -> dict[int, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Không tìm thấy class mapping: {path}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"class_mapping phải là object không rỗng: {path}")
    mapping: dict[int, str] = {}
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Nhãn không phải số nguyên: {key!r}") from exc
        if index in mapping or not isinstance(value, str) or not value.strip():
            raise ValueError(f"Mapping không hợp lệ tại nhãn {key!r}")
        mapping[index] = value.strip()
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError(f"class_mapping phải liên tục từ 0; nhận {sorted(mapping)}")
    if len({name.casefold() for name in mapping.values()}) != len(mapping):
        raise ValueError("Tên lớp trong mapping bị trùng")
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


def load_predictions(path: Path, mapping: dict[int, str], expected_split: str | None = None) -> list[dict[str, Any]]:
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
        split_column = next((lookup[name] for name in SPLIT_COLUMNS if name in lookup), None)
        rows, seen = [], set()
        for row_number, row in enumerate(reader, start=2):
            sample_id = str(row[id_column]).strip() if id_column and row.get(id_column) is not None else str(row_number - 2)
            if not sample_id:
                raise ValueError(f"Mẫu rỗng ở dòng {row_number} trong {path}")
            if sample_id in seen:
                raise ValueError(f"Mẫu trùng {sample_id!r} ở dòng {row_number} trong {path}")
            seen.add(sample_id)
            split = str(row[split_column]).strip() if split_column else ""
            if expected_split and split and split.casefold() != expected_split.casefold():
                raise ValueError(f"Split {split!r} ở dòng {row_number} không khớp {expected_split!r}")
            rows.append({"id": sample_id, "true": normalize_label(row[true_column], mapping, path),
                         "pred": normalize_label(row[pred_column], mapping, path), "split": split})
    if not rows:
        raise ValueError(f"Prediction CSV rỗng: {path}")
    return rows


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata and metadata[key] not in (None, ""):
            return metadata[key]
    return ""


def load_metadata(predictions_path: Path | None, explicit_path: Path | None) -> dict[str, Any]:
    candidate = explicit_path
    if candidate is None and predictions_path:
        name = predictions_path.name
        if name.endswith("_test_predictions.csv"):
            candidate = predictions_path.with_name(name.replace("_test_predictions.csv", "_metrics.json"))
        elif name.endswith("_predictions.csv"):
            candidate = predictions_path.with_name(name.replace("_predictions.csv", "_metrics.json"))
    if candidate is None or not candidate.is_file():
        return {}
    data = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metadata phải là object: {candidate}")
    data["_metadata_file"] = str(candidate)
    return data


def _validate_score(value: Any, key: str, source: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metric {key} không phải số trong {source}") from exc
    if not 0 <= number <= 1:
        raise ValueError(f"Metric {key} ngoài [0,1] trong {source}: {number}")
    return number


def _validate_count(value: Any, key: str, source: Path, *, allow_zero: bool = True) -> int:
    """Đọc một số đếm nguyên, không âm từ artifact do chủ model bàn giao."""
    if isinstance(value, bool):
        raise ValueError(f"{key} phải là số nguyên trong {source}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} phải là số nguyên trong {source}") from exc
    if str(value).strip() not in {str(number), f"+{number}"} if isinstance(value, str) else False:
        raise ValueError(f"{key} phải là số nguyên trong {source}")
    if number < 0 or (not allow_zero and number == 0):
        qualifier = "dương" if not allow_zero else "không âm"
        raise ValueError(f"{key} phải là số {qualifier} trong {source}")
    return number


def metrics_from_json(path: Path, mapping: dict[int, str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy metrics JSON: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metrics phải là object: {path}")
    missing = [key for key in (*SCORE_KEYS, "n_samples") if key not in data]
    if missing:
        raise ValueError(f"Metrics thiếu trường {missing} trong {path}")
    per_class = data.get("per_class")
    if not isinstance(per_class, dict):
        raise ValueError(f"Metrics thiếu per_class trong {path}")
    result = {"n_samples": _validate_count(data["n_samples"], "n_samples", path, allow_zero=False),
              **{key: _validate_score(data[key], key, path) for key in SCORE_KEYS},
              "per_class": {}, "confusion_matrix": data.get("confusion_matrix"),
              "metadata": {**data, "_metadata_file": str(path)}}
    for _, class_name in mapping.items():
        values = per_class.get(class_name)
        if not isinstance(values, dict) or any(metric not in values for metric in ("precision", "recall", "f1", "support")):
            raise ValueError(f"Metrics thiếu per_class[{class_name!r}] trong {path}")
        result["per_class"][class_name] = {metric: _validate_score(values[metric], metric, path)
                                            for metric in ("precision", "recall", "f1")}
        result["per_class"][class_name]["support"] = _validate_count(values["support"], f"support[{class_name}]", path)
    if sum(values["support"] for values in result["per_class"].values()) != result["n_samples"]:
        raise ValueError(f"Tổng support không khớp n_samples trong {path}")
    matrix = result["confusion_matrix"]
    if matrix is not None:
        if not isinstance(matrix, list) or len(matrix) != len(mapping) or any(not isinstance(row, list) or len(row) != len(mapping) for row in matrix):
            raise ValueError(f"confusion_matrix sai kích thước trong {path}")
        result["confusion_matrix"] = [[_validate_count(value, "confusion_matrix", path) for value in row] for row in matrix]
        if sum(sum(row) for row in result["confusion_matrix"]) != result["n_samples"]:
            raise ValueError(f"Tổng confusion_matrix không khớp n_samples trong {path}")
        for index, class_name in mapping.items():
            if sum(result["confusion_matrix"][index]) != result["per_class"][class_name]["support"]:
                raise ValueError(f"Hàng confusion_matrix của {class_name} không khớp support trong {path}")
    return result


def evaluate(rows: list[dict[str, Any]], mapping: dict[int, str]) -> dict[str, Any]:
    labels = list(mapping)
    y_true, y_pred = [row["true"] for row in rows], [row["pred"] for row in rows]
    if precision_recall_fscore_support is not None:
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    else:  # Fallback keeps schema checks usable in a minimal environment.
        matrix = [[0 for _ in labels] for _ in labels]
        for actual, predicted in zip(y_true, y_pred):
            matrix[actual][predicted] += 1
        precision, recall, f1, support = [], [], [], []
        for index in labels:
            tp = matrix[index][index]
            fp = sum(row[index] for row in matrix) - tp
            fn = sum(matrix[index]) - tp
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            precision.append(p); recall.append(r); f1.append(2 * p * r / (p + r) if p + r else 0.0); support.append(sum(matrix[index]))
    accuracy = (float(accuracy_score(y_true, y_pred)) if accuracy_score is not None
                else sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) / len(rows))
    return {"n_samples": len(rows), "accuracy": accuracy,
            "macro_precision": float(sum(precision) / len(precision)), "macro_recall": float(sum(recall) / len(recall)),
            "macro_f1": float(sum(f1) / len(f1)),
            "per_class": {mapping[index]: {"precision": float(precision[pos]), "recall": float(recall[pos]),
                                            "f1": float(f1[pos]), "support": int(support[pos])}
                          for pos, index in enumerate(labels)},
            "confusion_matrix": matrix}


def validate_same_test_set(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> None:
    first_ids, second_ids = {row["id"] for row in first}, {row["id"] for row in second}
    if first_ids != second_ids:
        raise ValueError("Hai prediction CSV không dùng cùng Test set; "
                         f"thiếu ở CNN={sorted(second_ids - first_ids)[:5]}, thiếu ở ResNet18={sorted(first_ids - second_ids)[:5]}")
    truth = {row["id"]: row["true"] for row in first}
    mismatch = [row["id"] for row in second if truth[row["id"]] != row["true"]]
    if mismatch:
        raise ValueError(f"Ground truth khác nhau giữa hai CSV (ví dụ: {mismatch[:5]})")


def validate_metrics_consistency(computed: dict[str, Any], reported: dict[str, Any], source: Path, tolerance: float = 1e-6) -> None:
    """Đối chiếu metrics JSON do chủ model bàn giao với prediction CSV."""
    for key in SCORE_KEYS:
        if key in reported and abs(float(reported[key]) - float(computed[key])) > tolerance:
            raise ValueError(f"Metric {key} trong {source} không khớp prediction CSV")
    reported_classes = reported.get("per_class")
    if isinstance(reported_classes, dict):
        for class_name, values in computed["per_class"].items():
            if class_name in reported_classes and isinstance(reported_classes[class_name], dict):
                if "f1" in reported_classes[class_name] and abs(float(reported_classes[class_name]["f1"]) - values["f1"]) > tolerance:
                    raise ValueError(f"Per-class F1 của {class_name} trong {source} không khớp prediction CSV")


def plot_confusion(metrics: dict[str, Any], mapping: dict[int, str], path: Path, title: str) -> None:
    if metrics.get("confusion_matrix") is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        # Giữ được bằng chứng confusion matrix trong môi trường tối giản;
        # khi có matplotlib, phía trên vẫn sinh PNG như quy ước báo cáo.
        svg = path.with_suffix(".svg")
        names = list(mapping.values())
        cells = []
        for i, row in enumerate(metrics["confusion_matrix"]):
            for j, value in enumerate(row):
                cells.append(f'<text x="{150 + j * 120}" y="{80 + i * 45}" text-anchor="middle">{value}</text>')
        svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="700" height="280">'
            '<text x="350" y="25" text-anchor="middle">Confusion matrix - ' + title + '</text>'
            + ''.join(cells) + '<text x="350" y="260" text-anchor="middle">columns: '
            + ', '.join(names) + '</text></svg>', encoding="utf-8")
        return
    matrix, names = metrics["confusion_matrix"], list(mapping.values())
    fig, ax = plt.subplots(figsize=(7, 5.5)); image = ax.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right"); ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(title)
    peak = max((max(row) for row in matrix), default=0) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > peak / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def comparison_row(name: str, split: str, metrics: dict[str, Any] | None, metadata: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"model": name, "status": "computed" if metrics else "pending", "split": split,
                           "n_samples": metrics.get("n_samples", "") if metrics else ""}
    for key in SCORE_KEYS:
        row[key] = metrics.get(key, "") if metrics else ""
    row.update({"total_parameters": _metadata_value(metadata, "total_parameters", "num_parameters", "parameters"),
                "trainable_parameters": _metadata_value(metadata, "trainable_parameters"),
                "train_seconds": _metadata_value(metadata, "train_seconds", "training_seconds"),
                "inference_seconds": _metadata_value(metadata, "inference_seconds", "test_seconds"),
                "device": _metadata_value(metadata, "device"), "checkpoint": _metadata_value(metadata, "checkpoint"),
                "checkpoint_epoch": _metadata_value(metadata, "checkpoint_epoch", "epoch"),
                "checkpoint_val_macro_f1": _metadata_value(metadata, "checkpoint_val_macro_f1", "val_macro_f1"),
                "checkpoint_val_loss": _metadata_value(metadata, "checkpoint_val_loss", "val_loss"),
                "seed": _metadata_value(metadata, "seed"),
                "python_version": _metadata_value(metadata, "python_version"),
                "torch_version": _metadata_value(metadata, "torch_version"),
                "torchvision_version": _metadata_value(metadata, "torchvision_version"),
                "metadata_file": _metadata_value(metadata, "_metadata_file")})
    for class_name in (metrics or {}).get("per_class", {}):
        row[f"{class_name}_f1"] = metrics["per_class"][class_name]["f1"]
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_analysis(path: Path, mapping: dict[int, str], split: str, metrics_by_model: dict[str, dict[str, Any] | None], metadata_by_model: dict[str, dict[str, Any]], files: dict[str, Path | None]) -> None:
    available = {name: score for name, score in metrics_by_model.items() if score}
    lines = ["# Phân tích đánh giá NV5", "", f"- Split: `{split}`; Test chỉ được dùng sau khi chốt checkpoint bằng Validation.", f"- Tạo lúc: `{datetime.now(timezone.utc).isoformat()}`.", f"- Python: `{platform.python_version()}`.", f"- Mapping: `{', '.join(mapping.values())}`.", "", "## Nguồn truy vết", ""]
    for model in metrics_by_model:
        source = files.get(model)
        lines.append(f"- **{model}**: `{source}`" if source else f"- **{model}**: chưa bàn giao artifact, trạng thái `pending`.")
        if metadata_by_model[model].get("_metadata_file"):
            lines.append(f"  - metadata: `{metadata_by_model[model]['_metadata_file']}`")
    lines += ["", "## Kết quả", "", "| Mô hình | Trạng thái | Số mẫu | Accuracy | Macro Precision | Macro Recall | Macro F1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for model, metrics in metrics_by_model.items():
        if metrics:
            lines.append(f"| {model} | computed | {metrics['n_samples']} | {metrics['accuracy']:.4f} | {metrics['macro_precision']:.4f} | {metrics['macro_recall']:.4f} | {metrics['macro_f1']:.4f} |")
        else:
            lines.append(f"| {model} | pending | — | — | — | — | — |")
    if available:
        best = max(available, key=lambda name: available[name]["macro_f1"])
        lines += ["", f"Macro F1 cao nhất trong các artifact hiện có: **{best}** ({available[best]['macro_f1']:.4f}). Đây không phải quyết định chọn lại checkpoint."]
    pair_counts: Counter[tuple[str, str]] = Counter()
    for metrics in available.values():
        matrix = metrics.get("confusion_matrix")
        if matrix:
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    if i != j: pair_counts[(mapping[i], mapping[j])] += int(value)
    lines += ["", "## Cặp lớp dễ nhầm", ""]
    lines.extend([f"- {actual} → {predicted}: {count} mẫu" for (actual, predicted), count in pair_counts.most_common() if count][:10] or ["- Chưa có đủ confusion matrix để phân tích."])
    lines += ["", "## Hạn chế và dữ liệu thiếu", "", "- PlantVillage là dữ liệu có kiểm soát; không suy rộng trực tiếp ra ảnh ngoài thực địa.", "- Ảnh ngoài bốn lớp vẫn bị ép vào một nhãn; chưa có đánh giá bệnh mới chớm.", "- ResNet18 chỉ được ghi nhận khi NV3 bàn giao metrics/predictions thật; không tạo số liệu thay thế.", "- Không dùng Test để chọn hyperparameter hoặc checkpoint.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _model_artifact(name: str, predictions: str | None, metrics_path: str | None, mapping: dict[int, str], split: str) -> tuple[dict[str, Any] | None, dict[str, Any], Path | None, list[dict[str, Any]] | None]:
    pred_path = Path(predictions) if predictions else None
    explicit_metrics = Path(metrics_path) if metrics_path else None
    metadata = load_metadata(pred_path, explicit_metrics)
    rows = load_predictions(pred_path, mapping, split) if pred_path else None
    metrics = evaluate(rows, mapping) if rows else None
    source = pred_path
    if metrics is None and explicit_metrics:
        metrics = metrics_from_json(explicit_metrics, mapping)
        source = explicit_metrics
    elif metrics and metadata:
        validate_metrics_consistency(metrics, metadata, Path(metadata["_metadata_file"]))
    if metrics and metadata.get("split") and str(metadata["split"]).casefold() != split.casefold():
        raise ValueError(f"Metadata của {name} ghi split={metadata['split']!r}, không khớp --split {split!r}")
    if metrics and metadata.get("n_samples") is not None and metrics.get("n_samples") not in ("", int(metadata["n_samples"])):
        raise ValueError(f"Số mẫu metadata của {name} không khớp artifact")
    return metrics, metadata, source, rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    mapping = load_class_mapping(Path(args.class_mapping))
    if not (args.cnn_predictions or args.cnn_metrics):
        raise ValueError("Cần --cnn-predictions hoặc --cnn-metrics cho CNN")
    cnn, cnn_meta, cnn_source, cnn_rows = _model_artifact("CNN", args.cnn_predictions, args.cnn_metrics, mapping, args.split)
    resnet, resnet_meta, resnet_source, resnet_rows = _model_artifact("ResNet18", args.resnet18_predictions, args.resnet18_metrics, mapping, args.split)
    if cnn_rows and resnet_rows:
        validate_same_test_set(cnn_rows, resnet_rows)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "comparison.csv", [comparison_row("CNN", args.split, cnn, cnn_meta), comparison_row("ResNet18", args.split, resnet, resnet_meta)])
    if cnn: plot_confusion(cnn, mapping, out / "cnn_confusion_matrix.png", f"CNN - {args.split}")
    if resnet: plot_confusion(resnet, mapping, out / "resnet18_confusion_matrix.png", f"ResNet18 - {args.split}")
    write_analysis(out / "analysis.md", mapping, args.split, {"CNN": cnn, "ResNet18": resnet}, {"CNN": cnn_meta, "ResNet18": resnet_meta}, {"CNN": cnn_source, "ResNet18": resnet_source})
    return {"output_dir": str(out), "models": {"CNN": "computed" if cnn else "pending", "ResNet18": "computed" if resnet else "pending"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cnn-predictions"); parser.add_argument("--resnet18-predictions")
    parser.add_argument("--cnn-metrics"); parser.add_argument("--resnet18-metrics")
    parser.add_argument("--class-mapping", default=str(DEFAULT_MAPPING)); parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    sys.exit(main())
