"""Dự đoán một ảnh lá cà chua bằng checkpoint CNN.

Phụ trách: NV2 - Nguyễn Văn Duy.

Chạy từ thư mục gốc dự án:
    python -m scripts.infer_cnn duong_dan_anh.jpg [--checkpoint checkpoints/cnn_best.pt]
In ra lớp dự đoán và xác suất softmax của 4 lớp (thứ tự theo class_mapping.json).
Xác suất là độ tin của mô hình, không phải chẩn đoán chắc chắn.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

from src.cnn_training import load_checkpoint, load_class_names, use_utf8_output

REPO_ROOT = Path(__file__).resolve().parents[1]


def predict_file(path, model, transform, class_names):
    with Image.open(path) as image:
        batch = transform(image.convert("RGB")).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        probs = torch.softmax(model(batch), dim=1)[0].tolist()
    index = max(range(len(probs)), key=probs.__getitem__)
    return {"index": index, "label": class_names[index],
            "probabilities": dict(zip(class_names, probs))}


def main(argv=None):
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image")
    parser.add_argument("--config", default="configs/cnn.yaml")
    parser.add_argument("--checkpoint")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    class_names = load_class_names(REPO_ROOT / config["data"]["class_mapping"])
    model, _ = load_checkpoint(Path(args.checkpoint or REPO_ROOT / config["checkpoint"]["path"]), class_names)
    try:
        from src.data.transforms import build_val_transform
    except ImportError as exc:
        raise SystemExit(f"Chưa có module dữ liệu của NV1 (Gate 1): {exc}") from exc
    result = predict_file(args.image, model, build_val_transform("cnn", config), class_names)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
