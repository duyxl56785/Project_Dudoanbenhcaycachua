"""NV2: kiểm tra tiện ích train/metrics/checkpoint trên tensor ngẫu nhiên.
Không phải test độ chính xác phân loại bệnh."""
import json

import pytest
import torch
from torch import nn

from tests.nv2_helpers import NAMES, synthetic_loader
from src.cnn_training import (EarlyStopping, compute_metrics, load_checkpoint, load_class_names,
                              run_epoch, save_checkpoint, set_seed)
from src.models.cnn import build_model


def test_compute_metrics_known_values():
    m = compute_metrics([0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 1, 1], NAMES)
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 1 + 1) / 4)
    assert m["macro_precision"] == pytest.approx((1 + 2 / 3 + 1 + 1) / 4)
    assert m["macro_recall"] == pytest.approx((0.5 + 1 + 1 + 1) / 4)
    assert m["confusion_matrix"] == [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert list(m["per_class"]) == NAMES
    assert m["per_class"]["Healthy"]["support"] == 2


def test_compute_metrics_counts_absent_predicted_class_as_zero():
    m = compute_metrics([0, 1, 2, 3], [0, 0, 0, 0], NAMES)
    assert m["per_class"]["Bacterial Spot"]["f1"] == 0.0


def test_early_stopping_max_mode():
    es = EarlyStopping(patience=2)
    assert [es.step(v) for v in (0.5, 0.6, 0.6, 0.55)] == [True, True, False, False]
    assert es.should_stop and es.best == 0.6 and es.best_epoch == 1


def test_same_seed_same_first_epoch_loss():
    losses = []
    for _ in range(2):
        set_seed(42)
        model, loader = build_model(4), synthetic_loader(16, seed=0)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        losses.append(run_epoch(model, loader, nn.CrossEntropyLoss(), "cpu", opt)["loss"])
    assert losses[0] == losses[1]


def test_run_epoch_without_optimizer_leaves_weights_unchanged():
    model = build_model(4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = run_epoch(model, synthetic_loader(8), nn.CrossEntropyLoss(), "cpu")
    assert set(out) == {"loss", "accuracy", "macro_f1"}
    assert not model.training
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_load_class_names_keeps_numeric_order(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"1": "B", "0": "A", "3": "D", "2": "C"}), encoding="utf-8")
    assert load_class_names(path) == ["A", "B", "C", "D"]
    path.write_text(json.dumps({"0": "A", "1": "A", "2": "C", "3": "D"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_class_names(path)


META = {"labels": NAMES, "preprocessing": {"image_size": 224}, "epoch": 3, "val_macro_f1": 0.5,
        "seed": 42, "config": {}, "git_commit": "test", "environment": {}}


def test_checkpoint_is_plain_state_dict(tmp_path):
    path = tmp_path / "cnn_best.pt"
    save_checkpoint(build_model(4), path, META)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and state
    assert all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items())
    build_model(4).load_state_dict(state, strict=True)


def test_load_rejects_label_mismatch(tmp_path):
    path = tmp_path / "cnn_best.pt"
    save_checkpoint(build_model(4), path, META)
    model, meta = load_checkpoint(path, NAMES)
    assert not model.training and meta["labels"] == NAMES
    with pytest.raises(ValueError):
        load_checkpoint(path, list(reversed(NAMES)))


def test_early_stopping_tie_prefers_lower_val_loss():
    es = EarlyStopping(patience=3)
    assert es.step(0.6, tiebreak=1.0) is True
    assert es.step(0.6, tiebreak=0.9) is True    # bằng F1, loss thấp hơn -> chọn epoch này
    assert es.step(0.6, tiebreak=0.95) is False  # bằng F1, loss cao hơn -> giữ epoch trước
    assert es.best_epoch == 1
