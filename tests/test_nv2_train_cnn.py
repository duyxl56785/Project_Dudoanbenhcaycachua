"""NV2: chạy thử vòng huấn luyện trên tensor ngẫu nhiên (2 epoch).
Không phải kết quả huấn luyện thật."""
import csv
import json

import pytest

from tests.nv2_helpers import NAMES, load_yaml, synthetic_loader
from scripts.train_cnn import plot_curves, train
from src.models.cnn import build_model


def test_train_on_synthetic_data_writes_artifacts(tmp_path):
    cfg = load_yaml("configs/cnn.yaml")
    cfg["training"]["epochs"] = 2
    cfg["data"]["test_split"] = str(tmp_path / "KHONG_TON_TAI.csv")  # Test không được đụng tới
    summary = train(build_model(4), synthetic_loader(16), synthetic_loader(8), cfg,
                    tmp_path, tmp_path / "cnn_best.pt", "cpu")
    with (tmp_path / "train_history.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert summary["epochs_run"] == 2 and len(rows) == 2 and "val_macro_f1" in rows[0]
    assert summary["best_epoch"] in (1, 2)
    assert (tmp_path / "cnn_best.pt").is_file()
    meta = json.loads((tmp_path / "cnn_best.meta.json").read_text(encoding="utf-8"))
    assert meta["labels"] == NAMES
    assert meta["epoch"] == summary["best_epoch"]
    assert meta["val_macro_f1"] == summary["best_val_macro_f1"]


def test_plot_curves_writes_png(tmp_path):
    history = tmp_path / "train_history.csv"
    history.write_text(
        "epoch,train_loss,train_accuracy,train_macro_f1,val_loss,val_accuracy,val_macro_f1,lr\n"
        "1,1.3,0.3,0.2,1.2,0.35,0.25,0.001\n2,1.1,0.4,0.3,1.15,0.4,0.3,0.001\n",
        encoding="utf-8")
    plot_curves(history, tmp_path / "curves.png")
    assert (tmp_path / "curves.png").stat().st_size > 0


def _has_nv1_dataset():
    try:
        from src.data.dataset import ManifestDataset  # noqa: F401
    except ImportError:
        return False
    return True


def test_use_utf8_output_keeps_vietnamese_readable_in_pipes():
    import subprocess
    import sys
    code = "from src.cnn_training import use_utf8_output; use_utf8_output(); import sys; sys.stderr.write('Chưa có dữ liệu')"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert proc.stderr.decode("utf-8") == "Chưa có dữ liệu"


@pytest.mark.skipif(_has_nv1_dataset(), reason="Nhánh đã có code dữ liệu của NV1")
def test_cli_without_nv1_data_module_exits_with_readable_message():
    import subprocess
    import sys
    proc = subprocess.run([sys.executable, "-m", "scripts.train_cnn", "--run-name", "khong_dung"],
                          capture_output=True)
    assert proc.returncode != 0
    assert "Chưa có module dữ liệu của NV1" in proc.stderr.decode("utf-8")
