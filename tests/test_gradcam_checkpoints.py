"""Optional acceptance checks using local approved weights and all eight demo images."""
import csv
import hashlib
import io
import json

from PIL import Image
import pytest
import torch

from demo.session import predict_uploads
from src.explainability import explain_image
from src.inference import ROOT, MODELS, _cached_model, class_names, predict_image


class Upload:
    def __init__(self, path):
        self.name, self.data = path.name, path.read_bytes()

    def getvalue(self):
        return self.data


def test_real_checkpoint_comparison():
    paths = sorted((ROOT / "demo_test_images").glob("*.JPG"))
    if not paths or not all((ROOT / "checkpoints" / name).is_file() for name in MODELS.values()):
        pytest.skip("Requires both local approved checkpoints and demo_test_images")
    uploads = [Upload(paths[0])]
    off, _ = predict_uploads(uploads, ("cnn", "resnet18"), predict_image, device="cpu")
    on, records = predict_uploads(uploads, ("cnn", "resnet18"), predict_image, device="cpu", explain_fn=explain_image)
    assert len(records) == 2 and all(record["status"] == "Thành công" for record in records)
    for model in MODELS:
        assert off[0]["results"][model]["probabilities"] == on[0]["results"][model]["probabilities"]
        assert on[0]["attention"][model]["png"] is not None


@pytest.mark.parametrize("model", ("cnn", "resnet18"))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_real_checkpoint_gradcam_does_not_change_predictions(model, device, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware/runtime unavailable")
    checkpoint = ROOT / "checkpoints" / MODELS[model]
    paths = sorted((ROOT / "demo_test_images").glob("*.JPG"))
    if not checkpoint.is_file() or not paths:
        pytest.skip("Requires local approved checkpoint and demo_test_images")
    names = class_names()
    with (ROOT / "data/splits/test.csv").open(encoding="utf-8", newline="") as stream:
        labels = {row["path"].rsplit("/", 1)[-1]: names[int(row["label"])] for row in csv.DictReader(stream)}
    uploads = [Upload(path) for path in paths]
    invalid = Upload.__new__(Upload)
    invalid.name, invalid.data = "invalid.png", b"not an image"
    uploads = uploads[:7] + [invalid]  # Batch limit includes the invalid image.
    before, _ = predict_uploads(uploads, (model,), predict_image, device=device)
    on, records = predict_uploads(uploads, (model,), predict_image, device=device, explain_fn=explain_image)
    assert "image" in on[-1]["errors"]
    assert records[-1]["status"] == "Lỗi"
    loaded = _cached_model(model, ROOT, device, checkpoint.stat().st_mtime_ns)
    state = {name: value.clone() for name, value in loaded.state_dict().items()}
    report = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        off = predict_image(image, model, device=device)
        attention = explain_image(image, model, off["top_index"], device=device)
        after = predict_image(image, model, device=device)
        assert off["top_index"] == after["top_index"]
        assert off["probabilities"] == after["probabilities"]
        assert attention["png"] is not None
        with Image.open(io.BytesIO(attention["png"])) as overlay:
            assert overlay.size == (224, 224)
        if off["top_label"] == "Healthy":
            assert not attention["has_box"]
        (tmp_path / f"{path.stem}_{model}.png").write_bytes(attention["png"])
        expected = next(label for name, label in labels.items() if path.name.endswith(name))
        report.append({"image": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "expected_from_test_manifest": expected, "predicted": off["top_label"],
                       "probabilities": off["probabilities"], "max_probability_delta": 0.0,
                       "has_box": attention["has_box"]})
    after, _ = predict_uploads(uploads, (model,), predict_image, device=device)
    for a, b, c in zip(before[:-1], on[:-1], after[:-1]):
        assert a["results"][model]["probabilities"] == b["results"][model]["probabilities"] == c["results"][model]["probabilities"]
        assert b["attention"][model]["png"] is not None
    assert set(row["expected_from_test_manifest"] for row in report) == set(names)
    assert all(torch.equal(state[name], value) for name, value in loaded.state_dict().items())
    assert not loaded.training and all(p.grad is None for p in loaded.parameters())
    layer = loaded.features[-1] if model == "cnn" else loaded.layer4
    assert not layer._forward_hooks
    evidence = {"model": model, "device": device,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "images": report}
    (tmp_path / f"{model}_{device}.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"model": model, "device": device, "images": len(report), "max_probability_delta": 0.0,
                      "correct_sample_labels": sum(row["predicted"] == row["expected_from_test_manifest"] for row in report)}))
