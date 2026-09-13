"""Small semantic model shared by training and offline export.

The optional initialization is generic COCO/VOC segmentation, not an anatomical
checkpoint. An Endoscapes fine-tune must run before output is called anatomical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

ARCHITECTURE = "deeplabv3_mobilenet_v3_large"
MODEL_ID = "holospex-deeplabv3-mobilenetv3"
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


def build_model(num_classes: int = 7, pretrained: bool = False) -> nn.Module:
    """No download unless pretrained=True. Keep the auxiliary head in both paths.

    Torchvision's pretrained constructor requires its original 21-class head;
    replace the final convolutions AFTER loading those generic weights.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must include background and at least one foreground class.")
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = deeplabv3_mobilenet_v3_large(
        weights=weights, weights_backbone=None, aux_loss=True,
    )
    for head in (model.classifier, model.aux_classifier):
        if head is not None:
            previous = head[-1]
            head[-1] = nn.Conv2d(previous.in_channels, num_classes, kernel_size=1)
    return model


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU. Explicit requests never fall back."""
    name = str(device)
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    result = torch.device(name)
    if result.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Supported devices are auto, cpu, mps, and cuda[:index].")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return result


def freeze_batchnorm(model: nn.Module) -> None:
    """ASPP pools to 1x1: frozen BN permits small/singleton training batches."""
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()
            for parameter in layer.parameters():
                parameter.requires_grad_(False)


def load_checkpoint(path: Path, device: str | torch.device = "auto") -> tuple[nn.Module, dict[str, Any]]:
    """Load tensor/builtin-only checkpoints; never unpickle arbitrary objects."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported Holospex checkpoint format.")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Checkpoint architecture does not match this model adapter.")
    classes = checkpoint.get("classes")
    if not isinstance(classes, list) or len(classes) < 2 or [entry.get("index") for entry in classes] != list(range(len(classes))):
        raise ValueError("Checkpoint classes must be in contiguous training-index order.")
    size = checkpoint.get("input_size", {})
    if any(not isinstance(size.get(key), int) or isinstance(size[key], bool) or size[key] <= 0 for key in ("width", "height")):
        raise ValueError("Checkpoint must contain a positive integer input size.")
    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(resolve_device(device)).eval(), checkpoint
