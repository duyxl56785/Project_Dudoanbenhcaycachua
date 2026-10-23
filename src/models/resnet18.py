"""NV3 ResNet18 architecture and trainable-parameter policies."""
from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

WEIGHTS_NAME = "IMAGENET1K_V1"


def build_resnet18(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """Build ResNet18; use pretrained weights only when explicitly requested."""
    if num_classes != 4:
        raise ValueError("NV3 contract requires exactly four tomato-leaf classes")
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def configure_stage(model: nn.Module, stage: str, unfreeze_layers: tuple[str, ...] = ("layer4",)) -> None:
    """Stage A trains head only; stage B trains head and chosen backbone blocks."""
    if stage not in ("feature_extraction", "fine_tuning"):
        raise ValueError(f"Unknown stage: {stage}")
    allowed = {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4"}
    if stage == "fine_tuning" and (not unfreeze_layers or set(unfreeze_layers) - allowed):
        raise ValueError("fine_tuning unfreeze_layers must name supported backbone layers")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if stage == "fine_tuning":
        for name in unfreeze_layers:
            for parameter in getattr(model, name).parameters():
                parameter.requires_grad = True


def training_mode(model: nn.Module) -> None:
    """Use train mode while preventing frozen BatchNorm running-stat updates."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
