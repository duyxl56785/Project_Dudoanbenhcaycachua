"""Tiện ích huấn luyện, đánh giá và checkpoint cho CNN.

Phụ trách: NV2 - Nguyễn Văn Duy.

Metrics dùng cùng định nghĩa với NV3 (sklearn, labels=[0,1,2,3],
zero_division=0) để hai mô hình so sánh công bằng.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

LABELS = [0, 1, 2, 3]


def use_utf8_output():
    # Windows mặc định ghi cp1252 khi đầu ra đi qua pipe/file, làm hỏng tiếng Việt.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class_names(path):
    """Tên 4 lớp theo thứ tự nhãn 0..3 trong class_mapping.json."""
    with open(path, encoding="utf-8") as stream:
        mapping = json.load(stream)
    if set(mapping) != {str(i) for i in LABELS}:
        raise ValueError(f"{path}: cần đúng các khóa 0, 1, 2, 3")
    names = [mapping[str(i)] for i in LABELS]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: tên lớp bị trùng")
    return names


def compute_metrics(y_true, y_pred, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for name, p, r, f, s in zip(class_names, precision, recall, f1, support)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Một lượt qua loader. Có optimizer thì huấn luyện, không thì chỉ đánh giá."""
    training = optimizer is not None
    model.train(training)
    total_loss, y_true, y_pred = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            y_true += labels.tolist()
            y_pred += logits.argmax(1).tolist()
    if not y_true:
        raise ValueError("Loader rỗng")
    metrics = compute_metrics(y_true, y_pred, [str(i) for i in LABELS])
    return {"loss": total_loss / len(y_true), "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"]}


META_KEYS = ("labels", "preprocessing", "epoch", "val_macro_f1", "seed", "config",
             "git_commit", "environment")


def meta_path(checkpoint_path):
    return Path(checkpoint_path).with_suffix(".meta.json")


def save_checkpoint(model, path, meta):
    """Lưu state_dict thuần (đúng hợp đồng demo NV4) và metadata ở file .meta.json."""
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"meta thiếu khóa: {', '.join(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    meta_path(path).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_checkpoint(path, class_names, device="cpu"):
    """Nạp CNN từ checkpoint; báo lỗi nếu thứ tự nhãn khác class_mapping hiện tại."""
    from src.models.cnn import build_model

    meta = json.loads(meta_path(path).read_text(encoding="utf-8"))
    if meta.get("labels") != list(class_names):
        raise ValueError(f"Nhãn trong checkpoint {meta.get('labels')} khác class_mapping {list(class_names)}")
    model = build_model(len(class_names))
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return model.to(device).eval(), meta


class EarlyStopping:
    """Dừng khi chỉ số Validation không cải thiện sau `patience` epoch liên tiếp."""

    def __init__(self, patience, mode="max"):
        if mode not in ("max", "min"):
            raise ValueError("mode phải là 'max' hoặc 'min'")
        self.patience, self.mode = patience, mode
        self.best = self.best_epoch = None
        self.should_stop = False
        self._epoch, self._bad = -1, 0

    def step(self, value):
        self._epoch += 1
        improved = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if improved:
            self.best, self.best_epoch, self._bad = value, self._epoch, 0
        else:
            self._bad += 1
            self.should_stop = self._bad >= self.patience
        return improved
