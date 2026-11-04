"""NV2 x NV1: chạy train_cnn/eval_cnn với Dataset và transform thật của NV1
trên ảnh GIẢ (màu đơn sắc) theo đúng cấu trúc raw/color/<lớp>/.
Chỉ kiểm tra giao diện khớp nhau, không phải kết quả huấn luyện.
Tự bỏ qua khi nhánh hiện tại chưa có code dữ liệu của NV1."""
import csv
import json

import pytest
import yaml
from PIL import Image

dataset_module = pytest.importorskip("src.data.dataset")
if not hasattr(dataset_module, "ManifestDataset"):
    pytest.skip("Chưa có ManifestDataset của NV1", allow_module_level=True)

from scripts import eval_cnn, train_cnn  # noqa: E402

FOLDERS = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Bacterial_spot"]


def _fake_split(data_root, manifest, n_per_class):
    rows = []
    for label, folder in enumerate(FOLDERS):
        (data_root / "raw/color" / folder).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            rel = f"raw/color/{folder}/{manifest.stem}_{i}.JPG"
            Image.new("RGB", (256, 256), (40 * label, 100, 30 * i)).save(data_root / rel, "JPEG")
            rows.append({"path": rel, "label": label})
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("no_augmentation", [False, True])
def test_train_then_eval_with_nv1_dataset(tmp_path, no_augmentation):
    data_root = tmp_path / "data"
    _fake_split(data_root, tmp_path / "train.csv", 2)
    _fake_split(data_root, tmp_path / "val.csv", 1)
    with open("configs/cnn.yaml", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg["training"].update(epochs=1, batch_size=4)
    cfg["data"].update(train_split=str(tmp_path / "train.csv"), val_split=str(tmp_path / "val.csv"),
                       test_split=str(tmp_path / "KHONG_TON_TAI.csv"))
    cfg["output"]["results_dir"] = str(tmp_path / "results")
    config_path = tmp_path / "cnn.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    info = train_cnn.run(config_path, data_root, "smoke", no_augmentation=no_augmentation)
    run_dir = tmp_path / "results/runs/smoke"
    assert info["n_train"] == 8 and info["n_val"] == 4
    assert (run_dir / "cnn_best.pt").is_file() and (run_dir / "cnn_train_curves.png").is_file()

    report = eval_cnn.run(config_path, data_root, "val", checkpoint=run_dir / "cnn_best.pt")
    assert report["n_samples"] == 4
    with (tmp_path / "results/cnn_val_predictions.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert all("raw" in r["path"] and r["path"].endswith(".JPG") for r in rows)
    assert json.loads((tmp_path / "results/cnn_val_metrics.json").read_text(encoding="utf-8"))["split"] == "val"
