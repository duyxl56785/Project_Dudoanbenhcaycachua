"""Session-only helpers; image bytes are never stored in history."""
import csv
import io
import hashlib
from collections import Counter
from datetime import datetime

from src.inference import InferenceError, MAX_IMAGE_BYTES, read_image

MAX_UPLOADS = 8
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
        try:
            data = upload.getvalue()
            image_id = hashlib.sha256(data).hexdigest()
            if len(data) > MAX_IMAGE_BYTES:
                raise InferenceError("Ảnh vượt giới hạn 10 MiB.")
            image = read_image(data)
        except InferenceError as exc:
            item["errors"]["image"] = str(exc)
            continue
        for model in models:
            try:
                result = predict_fn(image, model, device=device)
                item["results"][model] = result
                records.append({"image_name": upload.name,
                                "image_id": image_id,
                                "predicted_at": now().astimezone().isoformat(timespec="seconds"),
                                "model": MODEL_LABELS[model], "top_label": result["top_label"],
                                "top_probability": result["probabilities"][result["top_label"]],
                                "probabilities": dict(result["probabilities"]),
                                "forward_ms": result["forward_ms"]})
            except InferenceError as exc:
                item["errors"][model] = str(exc)
    return items, records


def export_csv(records, names):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["Tên ảnh", "Mô hình", "Nhãn dự đoán", "Xác suất top-1",
                     *[f"Xác suất {name}" for name in names], "Thời gian dự đoán", "Forward (ms)"])
    for row in records:
        writer.writerow([row["image_name"], row["model"], row["top_label"],
                         f'{row["top_probability"]:.6f}',
                         *[f'{row["probabilities"][name]:.6f}' for name in names],
                         row["predicted_at"], f'{row["forward_ms"]:.2f}'])
    return ("\ufeff" + output.getvalue()).encode("utf-8")


def dashboard(records, names):
    by_model = Counter(row["model"] for row in records)
    by_label = Counter(row["top_label"] for row in records)
    return {"predictions": len(records), "images": len({row.get("image_id", row["image_name"]) for row in records}),
            "by_model": {name: by_model[name] for name in MODEL_LABELS.values()},
            "by_label": {name: by_label[name] for name in names},
            "mean_forward_ms": sum(row["forward_ms"] for row in records) / len(records) if records else None}
