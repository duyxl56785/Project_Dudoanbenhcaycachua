"""Session-only helpers; image bytes are never stored in history."""
import csv
import io
import hashlib
from collections import Counter
from datetime import datetime

from src.inference import InferenceError, MAX_IMAGE_BYTES, read_image

MAX_UPLOADS = 8
MAX_HISTORY = 200
MODEL_LABELS = {"cnn": "CNN", "resnet18": "ResNet18"}


def selected_models(choice):
    return {"CNN": ("cnn",), "ResNet18": ("resnet18",), "So sánh cả hai": ("cnn", "resnet18")}[choice]


def predict_uploads(uploads, models, predict_fn, *, device="auto", now=None):
    if len(uploads) > MAX_UPLOADS:
        raise InferenceError(f"Chỉ tải tối đa {MAX_UPLOADS} ảnh mỗi lượt.")
    now = now or datetime.now
    items, records = [], []
    for upload in uploads:
        item = {"name": upload.name, "results": {}, "errors": {}}
        items.append(item)
        image_id = None
        try:
            data = upload.getvalue()
            image_id = hashlib.sha256(data).hexdigest()
            if len(data) > MAX_IMAGE_BYTES:
                raise InferenceError("Ảnh vượt giới hạn 10 MiB.")
            image = read_image(data)
        except InferenceError as exc:
            item["errors"]["image"] = str(exc)
            records.append(_record(upload.name, image_id, "—", now, error=str(exc)))
            continue
        for model in models:
            try:
                result = predict_fn(image, model, device=device)
                item["results"][model] = result
                records.append(_record(upload.name, image_id, MODEL_LABELS[model], now, result=result))
            except InferenceError as exc:
                item["errors"][model] = str(exc)
                records.append(_record(upload.name, image_id, MODEL_LABELS[model], now, error=str(exc)))
    return items, records


def _record(name, image_id, model, now, *, result=None, error=None):
    return {"image_name": name, "image_id": image_id,
            "predicted_at": now().astimezone().isoformat(timespec="seconds"),
            "model": model, "top_label": result["top_label"] if result else "",
            "top_probability": result["probabilities"][result["top_label"]] if result else None,
            "probabilities": dict(result["probabilities"]) if result else {},
            "forward_ms": result["forward_ms"] if result else None,
            "status": "Thành công" if result else "Lỗi", "error": error or ""}


def append_history(history, records):
    """Keep only recent metadata; caller appends only on an explicit button click."""
    return (history + records)[-MAX_HISTORY:]


def _excel_safe(value):
    """Prevent spreadsheet formulas in user-controlled filenames or labels."""
    value = str(value)
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def export_csv(records, names):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["timestamp", "filename", "model", "predicted_class", "top1_probability",
                     "healthy_probability", "early_blight_probability", "late_blight_probability",
                     "bacterial_spot_probability", "status"])
    for row in records:
        probabilities = row.get("probabilities", {})
        writer.writerow([row["predicted_at"], _excel_safe(row["image_name"]), _excel_safe(row["model"]),
                         _excel_safe(row.get("top_label", "")),
                         f'{row["top_probability"]:.6f}' if row.get("top_probability") is not None else "",
                         *[f'{probabilities[name]:.6f}' if name in probabilities else "" for name in names],
                         row.get("status", "Thành công")])
    return ("\ufeff" + output.getvalue()).encode("utf-8")


def dashboard(records, names, model=None):
    filtered = [row for row in records if model is None or row["model"] == model]
    successes = [row for row in filtered if row.get("status", "Thành công") == "Thành công"]
    by_model = Counter(row["model"] for row in successes)
    by_label = Counter(row["top_label"] for row in successes)
    return {"predictions": len(successes), "errors": len(filtered) - len(successes),
            "images": len({row.get("image_id") or row["image_name"] for row in successes}),
            "by_model": {name: by_model[name] for name in MODEL_LABELS.values()},
            "by_label": {name: by_label[name] for name in names},
            "mean_forward_ms": sum(row["forward_ms"] for row in successes) / len(successes) if successes else None}
