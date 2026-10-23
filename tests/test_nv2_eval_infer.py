"""NV2: kiểm tra đánh giá và dự đoán một ảnh với mô hình chưa huấn luyện.
Không phải kết quả phân loại thật."""
import pytest
from PIL import Image

from nv2_helpers import NAMES, simple_transform, synthetic_loader
from scripts.eval_cnn import evaluate
from scripts.infer_cnn import predict_file
from src.models.cnn import build_model


@pytest.mark.parametrize("mode", ["RGBA", "L", "RGB"])
def test_predict_file_any_image_mode(tmp_path, mode):
    img = tmp_path / f"x_{mode}.png"
    Image.new(mode, (300, 200)).save(img)
    out = predict_file(img, build_model(4).eval(), simple_transform(128), NAMES)
    assert list(out["probabilities"]) == NAMES
    assert sum(out["probabilities"].values()) == pytest.approx(1, abs=1e-5)
    assert out["label"] == NAMES[out["index"]]


def test_evaluate_returns_one_row_per_sample():
    metrics, rows = evaluate(build_model(4).eval(), synthetic_loader(8, shuffle=False), NAMES, "cpu")
    assert [r["index"] for r in rows] == list(range(8))
    assert [r["true"] for r in rows] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert all(set(f"p_{n}" for n in NAMES) <= set(r) for r in rows)
    assert len(metrics["confusion_matrix"]) == 4
