"""NV2: kiểm tra hợp đồng của mô hình CNN (shape, số tham số, backward).
Không phải test độ chính xác phân loại bệnh."""
import pytest
import torch
from torch import nn

from src.models.cnn import build_model, count_parameters


def test_output_shape_and_param_count():
    model = build_model(4).eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 128, 128)).shape == (2, 4)
    assert count_parameters(model) == 389_924


def test_rejects_other_class_counts():
    with pytest.raises(ValueError):
        build_model(3)


def test_one_batch_backward_updates_weights():
    torch.manual_seed(0)
    model = build_model(4).train()
    before = model.classifier.weight.detach().clone()
    loss = nn.CrossEntropyLoss()(model(torch.randn(4, 3, 128, 128)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    torch.optim.Adam(model.parameters(), 1e-3).step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.classifier.weight)


def test_eval_mode_is_deterministic():
    model = build_model(4).eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        assert torch.equal(model(x), model(x))
