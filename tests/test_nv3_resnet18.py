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


def test_manifest_mapping_and_transforms(tmp_path):
    labels = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot"]
    image = tmp_path / "leaf.png"
    Image.new("RGB", (300, 300), color=(12, 80, 40)).save(image)
    csv_path = tmp_path / "val.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label"])
        writer.writeheader(); writer.writerow({"path": "leaf.png", "label": "3"})
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
