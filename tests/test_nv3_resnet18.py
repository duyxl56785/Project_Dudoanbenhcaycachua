"""CPU-only NV3 contract tests. These are NOT plant-disease accuracy tests."""
import csv
import json

import pytest
import torch
from PIL import Image

from src.models.resnet18 import build_resnet18, configure_stage, parameter_counts, training_mode
from src.resnet18_workflow import ManifestDataset, calculate_metrics, image_transform, read_config, read_labels


def test_resnet18_output_and_freezing():
    model = build_resnet18(4, pretrained=False)  # never download pretrained weights during smoke tests
    assert model.fc.out_features == 4
    configure_stage(model, "feature_extraction")
    assert all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith("fc."))
    assert all(p.requires_grad for p in model.fc.parameters())
    assert parameter_counts(model)["trainable"] == 2052
    training_mode(model)
    assert not model.bn1.training
    assert model.fc.training
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 4)


def test_finetuning_unfreezes_only_layer4_and_head():
    model = build_resnet18(4, pretrained=False)
    configure_stage(model, "fine_tuning", ("layer4",))
    assert all(p.requires_grad for p in model.layer4.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())
    assert all(not p.requires_grad for p in model.layer3.parameters())
    training_mode(model)
    assert not model.layer3[0].bn1.training
    assert model.layer4[0].bn1.training


@pytest.fixture
def shared_nv1():
    from src.data import dataset, transforms
    if not hasattr(dataset, "ManifestDataset") or not hasattr(transforms, "build_val_transform"):
        pytest.skip("NV1 PR #8 not integrated in this checkout; run combined-branch integration tests")


def test_manifest_mapping_and_transforms(tmp_path, shared_nv1):
    labels = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot"]
    image = tmp_path / "raw/color/Tomato___Bacterial_spot/leaf.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (300, 300), color=(12, 80, 40)).save(image)
    csv_path = tmp_path / "val.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label"])
        writer.writeheader(); writer.writerow({"path": "raw/color/Tomato___Bacterial_spot/leaf.png", "label": "3"})
    config = {"preprocessing": {"image_size": 224, "resize_size": 256,
                                 "normalize_mean": [0.485, 0.456, 0.406],
                                 "normalize_std": [0.229, 0.224, 0.225]},
              "augmentation": {"train": {"random_horizontal_flip": True, "color_jitter": 0.1}}}
    dataset = ManifestDataset(csv_path, tmp_path, labels, image_transform(config, train=False))
    tensor, label = dataset[0]
    assert tensor.shape == (3, 224, 224) and label == 3
    assert "Random" not in str(image_transform(config, train=False))
    with pytest.raises(FileNotFoundError):
        ManifestDataset(tmp_path / "missing.csv", tmp_path, labels, image_transform(config, train=False))


def test_metrics_four_class_shape():
    metrics = calculate_metrics([0, 1, 2, 3], [0, 1, 3, 3], ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot"])
    assert metrics["accuracy"] == 0.75
    assert len(metrics["confusion_matrix"]) == 4


def test_head_backprop_checkpoint_round_trip(tmp_path):
    from src.resnet18_workflow import checkpoint_payload, load_checkpoint
    config = read_config(__import__("pathlib").Path(__file__).resolve().parents[1] / "configs/resnet18.yaml")
    labels = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot"]
    model = build_resnet18(4, pretrained=False)
    configure_stage(model, "feature_extraction")
    training_mode(model)
    x = torch.randn(2, 3, 64, 64)
    before = model.fc.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()(model(x), torch.tensor([0, 3]))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.fc.weight.grad is not None
    assert all(p.grad is None for p in model.layer4.parameters())
    optimizer.step()
    assert not torch.equal(before, model.fc.weight)
    path = tmp_path / "resnet18_smoke.pt"
    torch.save(checkpoint_payload(model, config, labels, "feature_extraction", 0.0), path)
    restored, meta = load_checkpoint(path, config, labels, torch.device("cpu"))
    assert meta["stage"] == "feature_extraction"
    assert torch.equal(restored.fc.weight, model.fc.weight)
    with pytest.raises(ValueError):
        load_checkpoint(path, config, labels[::-1], torch.device("cpu"))


def test_nv4_offline_builder_and_plain_state_dict(tmp_path, monkeypatch):
    """NV4 can reconstruct architecture offline; no production conversion here."""
    from torchvision.models import ResNet18_Weights
    from src.models.resnet18 import build_model

    def deny_download(*args, **kwargs):
        raise AssertionError("Inference builder must not download weights")

    monkeypatch.setattr(ResNet18_Weights, "get_state_dict", deny_download)
    model = build_model(num_classes=4).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        expected = model(x)
    path = tmp_path / "plain_state.pt"
    torch.save(model.state_dict(), path)
    restored = build_model(num_classes=4)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(x), expected)
    with pytest.raises(ValueError, match="four"):
        build_model(num_classes=5)


@pytest.mark.parametrize("stage", ["feature_extraction", "fine_tuning"])
def test_frozen_batchnorm_buffers_stay_fixed_during_training(stage):
    model = build_resnet18(pretrained=False)
    configure_stage(model, stage, ("layer4",))
    training_mode(model)
    frozen = model.layer3[0].bn1
    before = {name: value.clone() for name, value in frozen.named_buffers()}
    active_count = model.layer4[0].bn1.num_batches_tracked.clone()
    loss = torch.nn.CrossEntropyLoss()(model(torch.randn(2, 3, 64, 64)), torch.tensor([0, 3]))
    loss.backward()
    for name, value in frozen.named_buffers():
        assert torch.equal(value, before[name])
    expected_count = active_count + (stage == "fine_tuning")
    assert torch.equal(model.layer4[0].bn1.num_batches_tracked, expected_count)
    assert all(p.grad is None for p in model.layer3.parameters())
    assert (model.layer4[0].conv1.weight.grad is not None) == (stage == "fine_tuning")


def test_plain_checkpoint_metadata_and_legacy_compatibility(tmp_path):
    from pathlib import Path
    from src.models.resnet18 import build_model
    from src.resnet18_workflow import checkpoint_payload, load_checkpoint, save_checkpoint
    config = read_config(Path(__file__).resolve().parents[1] / "configs/resnet18.yaml")
    labels = read_labels(Path(__file__).resolve().parents[1] / "configs/class_mapping.json")
    model = build_model()
    payload = checkpoint_payload(model, config, labels, "feature_extraction", 0.25)
    metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    path = tmp_path / "delivery.pt"
    save_checkpoint(path, model, metadata)
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert "model_state_dict" not in state
    build_model().load_state_dict(state, strict=True)
    restored, meta = load_checkpoint(path, config, labels, torch.device("cpu"))
    assert meta == metadata
    torch.testing.assert_close(restored.fc.weight, model.fc.weight)
    with pytest.raises(ValueError, match="mapping"):
        load_checkpoint(path, config, labels[::-1], torch.device("cpu"))
    import copy
    wrong = copy.deepcopy(config)
    wrong["preprocessing"]["image_size"] = 128
    with pytest.raises(ValueError, match="preprocessing"):
        load_checkpoint(path, wrong, labels, torch.device("cpu"))
    path.with_suffix(".meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata"):
        load_checkpoint(path, config, labels, torch.device("cpu"))


def test_synthetic_two_stage_train_eval_infer(tmp_path, monkeypatch, shared_nv1):
    """Exercise the real pipeline; synthetic metrics are never project results."""
    from pathlib import Path
    import yaml
    import scripts.train_resnet18 as training
    from scripts.eval_resnet18 import run as evaluate
    from scripts.infer_resnet18 import run as infer
    from src.models.resnet18 import build_model
    root = Path(__file__).resolve().parents[1]
    config = read_config(root / "configs/resnet18.yaml")
    config["data"]["class_mapping"] = str(root / "configs/class_mapping.json")
    folders = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Bacterial_spot"]
    for split in ("train", "val", "test"):
        manifest = tmp_path / f"{split}.csv"
        config["data"][f"{split}_split"] = str(manifest)
        with manifest.open("w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(["path", "label", "group"])
            for label, folder in enumerate(folders):
                path = Path("raw/color") / folder / f"{split}.png"
                (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (72, 72), (label * 40, 100, 30)).save(tmp_path / path)
                writer.writerow([path.as_posix(), label, f"{split}-{label}"])
    config["preprocessing"].update(image_size=64, resize_size=72)
    config["training"]["batch_size"] = 4
    for stage in config["stages"].values():
        stage["epochs"] = 1
    config["output"] = {"results_dir": str(tmp_path / "results"), "figures_dir": str(tmp_path / "figures")}
    config["checkpoint"]["path"] = str(tmp_path / "delivery.pt")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(training, "build_resnet18", lambda **kwargs: build_model())
    real_dataset = training.ManifestDataset
    def train_val_only(manifest, *args, **kwargs):
        assert Path(manifest).name != "test.csv", "Training must never read Test"
        return real_dataset(manifest, *args, **kwargs)
    monkeypatch.setattr(training, "ManifestDataset", train_val_only)
    result = training.run(config_path, tmp_path)
    assert result["selected_stage"] in ("feature_extraction", "fine_tuning")
    history = list(csv.DictReader((tmp_path / "results/train_history.csv").open()))
    assert [r["stage"] for r in history] == ["feature_extraction", "fine_tuning"]
    assert (tmp_path / "delivery.meta.json").is_file()
    before = (tmp_path / "delivery.pt").read_bytes()
    with pytest.raises(FileExistsError, match="preserved"):
        training.run(config_path, tmp_path)
    assert (tmp_path / "delivery.pt").read_bytes() == before
    metrics = evaluate(config_path, tmp_path)
    assert len(metrics["confusion_matrix"]) == 4
    prediction = infer(tmp_path / "raw/color/Tomato___healthy/test.png", config_path)
    assert list(prediction["scores"]) == read_labels(root / "configs/class_mapping.json")
    assert sum(prediction["scores"].values()) == pytest.approx(1.0)


def test_nv4_actual_loader_predictions_match_nv3(tmp_path, shared_nv1):
    from pathlib import Path
    from src import inference
    if not hasattr(inference, "predict_image"):
        pytest.skip("NV4 inference API not integrated; run combined NV1/NV3/NV4 tests")
    from src.models.resnet18 import build_model
    from src.resnet18_workflow import checkpoint_payload, save_checkpoint, predict_image
    root = Path(__file__).resolve().parents[1]
    config = read_config(root / "configs/resnet18.yaml")
    labels = read_labels(root / "configs/class_mapping.json")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/resnet18.yaml").write_text((root / "configs/resnet18.yaml").read_text())
    (tmp_path / "configs/class_mapping.json").write_text(json.dumps(dict(enumerate(labels))))
    model = build_model().eval()
    metadata = checkpoint_payload(model, config, labels, "feature_extraction", 0.0)
    metadata.pop("model_state_dict")
    save_checkpoint(tmp_path / "checkpoints/resnet18_best.pt", model, metadata)
    image_path = tmp_path / "synthetic.png"
    Image.new("RGB", (300, 260), (50, 120, 30)).save(image_path)
    expected = predict_image(model, image_path, config, labels, torch.device("cpu"))
    with Image.open(image_path) as image:
        actual = inference.predict_image(image.convert("RGB"), "resnet18", root=tmp_path)
    assert actual["top_label"] == expected["label"]
    assert actual["probabilities"] == pytest.approx(expected["scores"])
    inference._cached_model.cache_clear()


def test_nv1_create_dataloaders_rgb_normalization_contract(tmp_path, shared_nv1):
    """NV1 public loader and NV3 wrapper must produce identical Val tensors."""
    from pathlib import Path
    from src.data.dataset import create_dataloaders
    from src.data.transforms import build_train_transform, build_val_transform
    root = Path(__file__).resolve().parents[1]
    config = read_config(root / "configs/resnet18.yaml")
    manifest_dir = tmp_path / "splits"
    manifest_dir.mkdir()
    folders = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Bacterial_spot"]
    for split in ("train", "val", "test"):
        with (manifest_dir / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "label", "group"])
            for label, folder in enumerate(folders):
                relative = Path("raw/color") / folder / f"{split}.png"
                path = tmp_path / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                # Grayscale input verifies conversion to RGB in the shared Dataset.
                Image.new("L", (300, 260), 128).save(path)
                writer.writerow([relative.as_posix(), label, f"{split}-{label}"])
    loaders, datasets, mapping = create_dataloaders(
        data_root=tmp_path, model_name="resnet18", config=config,
        manifest_dir=manifest_dir, batch_size=4, num_workers=0, seed=42)
    assert mapping == {name: index for index, name in enumerate(read_labels(root / "configs/class_mapping.json"))}
    images, targets = next(iter(loaders["train"]))
    assert images.shape == (4, 3, 224, 224)
    assert sorted(targets.tolist()) == [0, 1, 2, 3]
    first = datasets["val"][0][0]
    torch.testing.assert_close(first, datasets["val"][0][0], rtol=0, atol=0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(first[:, 0, 0], (128 / 255 - mean) / std)
    with Image.open(datasets["val"].samples[0][0]) as image:
        rgb = image.convert("RGB")
        torch.testing.assert_close(first, image_transform(config, train=False)(rgb), rtol=0, atol=0)
        torch.testing.assert_close(first, build_val_transform("resnet18", config)(rgb), rtol=0, atol=0)
    assert str(image_transform(config, train=True)) == str(build_train_transform("resnet18", config))
