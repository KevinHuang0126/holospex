"""Small semantic model shared by training and offline export.

The optional initialization is generic COCO/VOC segmentation, not an anatomical
checkpoint. An Endoscapes fine-tune must run before output is called anatomical.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
)

# Preserve these names/defaults for existing callers and format-v1 checkpoints.
ARCHITECTURE = "deeplabv3_mobilenet_v3_large"
MODEL_ID = "holospex-deeplabv3-mobilenetv3"
MODEL_IDS = {
    ARCHITECTURE: MODEL_ID,
    "deeplabv3_resnet50": "holospex-deeplabv3-resnet50",
    "deeplabv3plus_mobilenet_v3_large": "holospex-deeplabv3plus-mobilenetv3",
}
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


def _separable_decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Keep refinement at stride four affordable on the mobile backbone."""
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
        nn.GroupNorm(8, in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.ReLU(inplace=True),
    )


class _MobileNetDeepLabV3Plus(nn.Module):
    """DeepLabV3+ style detail decoder with an existing Torchvision encoder.

    This is a local MobileNet variant, not an official Torchvision pretrained
    DeepLabV3+ model. The backbone, ASPP and following context convolution retain
    their generic pretrained weights. The fresh stride-four decoder uses group
    normalization because training freezes the pretrained batch normalization.
    """

    def __init__(self, base_model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = base_model.backbone
        # Torchvision's dilated MobileNetV3-Large: block 3 ends its 24-channel,
        # stride-four stage; aux is block 4 (stride eight), out is block 16.
        self.backbone.return_layers["3"] = "low"
        self.classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])
        self.aux_classifier = base_model.aux_classifier
        low_channels = self.backbone["3"].out_channels
        self.low_projection = nn.Sequential(
            nn.Conv2d(low_channels, 48, 1, bias=False),
            nn.GroupNorm(8, 48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            _separable_decoder_block(256 + 48, 128),
            _separable_decoder_block(128, 128),
            nn.Conv2d(128, num_classes, 1),
        )
        # Initialize only new modules. In particular, leave pretrained context
        # weights and batch-normalization statistics exactly as loaded.
        for module in (self.low_projection, self.decoder):
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.normal_(self.decoder[-1].weight, std=0.01)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        low = self.low_projection(features["low"])
        context = self.classifier(features["out"])
        context = F.interpolate(context, size=low.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat((context, low), dim=1))
        result = {"out": F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)}
        if self.aux_classifier is not None:
            auxiliary = self.aux_classifier(features["aux"])
            result["aux"] = F.interpolate(auxiliary, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return result


def model_id_for_architecture(architecture: str) -> str:
    if not isinstance(architecture, str) or architecture not in MODEL_IDS:
        raise ValueError(f"Unsupported architecture {architecture!r}; choose from {sorted(MODEL_IDS)}.")
    return MODEL_IDS[architecture]


def build_model(num_classes: int = 7, pretrained: bool = False, *, architecture: str = ARCHITECTURE) -> nn.Module:
    """No download unless pretrained=True. Keep the auxiliary head in both paths.

    Torchvision's pretrained constructor requires its original 21-class head;
    replace the final convolutions AFTER loading those generic weights.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must include background and at least one foreground class.")
    model_id_for_architecture(architecture)
    constructor, weights_enum = {
        ARCHITECTURE: (deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights),
        "deeplabv3_resnet50": (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights),
        "deeplabv3plus_mobilenet_v3_large": (deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights),
    }[architecture]
    weights = weights_enum.DEFAULT if pretrained else None
    model = constructor(
        weights=weights, weights_backbone=None, aux_loss=True,
    )
    for head in (model.classifier, model.aux_classifier):
        if head is not None:
            previous = head[-1]
            head[-1] = nn.Conv2d(previous.in_channels, num_classes, kernel_size=1)
    if architecture == "deeplabv3plus_mobilenet_v3_large":
        return _MobileNetDeepLabV3Plus(model, num_classes)
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


def load_backbone_checkpoint(model: nn.Module, path: Path, *, architecture: str) -> dict[str, Any]:
    """Initialize only a ResNet50 backbone from a safe, self-contained artifact.

    Validate every tensor before mutating the model. The generic segmentation
    heads and their normalization statistics remain untouched. Source metadata
    describes the supplied pretraining artifact; this is not anatomy training.
    """
    if architecture != "deeplabv3_resnet50":
        raise ValueError("Backbone initialization is supported only for deeplabv3_resnet50.")
    checkpoint_path = Path(path).resolve()

    def digest() -> str:
        result = hashlib.sha256()
        with checkpoint_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(chunk)
        return result.hexdigest()

    size, before = checkpoint_path.stat().st_size, digest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or type(checkpoint.get("format_version")) is not int
            or checkpoint.get("format_version") != 1
            or checkpoint.get("artifact_type") != "holospex_pretrained_backbone"
            or checkpoint.get("architecture") != "resnet50"):
        raise ValueError("Unsupported pretrained backbone artifact format or architecture.")
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Backbone artifact requires embedded JSON source provenance.")
    try:
        provenance = json.loads(json.dumps(provenance, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Backbone provenance must contain only finite JSON metadata.") from error
    state = checkpoint.get("backbone_state")
    if not isinstance(state, dict):
        raise ValueError("Backbone artifact must contain a tensor state dictionary.")
    expected = model.backbone.state_dict()
    if set(state) != set(expected):
        raise ValueError("Backbone tensor keys must exactly match the model; missing or unexpected keys found.")
    for key, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.device.type != "cpu" or tensor.shape != expected[key].shape
                or tensor.dtype != expected[key].dtype):
            raise ValueError(f"Backbone tensor has incompatible shape, dtype, or storage: {key}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Backbone tensor contains nonfinite values: {key}")
    if checkpoint_path.stat().st_size != size or digest() != before:
        raise RuntimeError("Backbone checkpoint changed while it was being validated.")
    model.backbone.load_state_dict(state, strict=True)
    return {"architecture": architecture, "path": str(checkpoint_path), "sha256": before,
            "bytes": size, "tensor_count": len(state), "source_provenance": provenance}


def load_checkpoint(path: Path, device: str | torch.device = "auto") -> tuple[nn.Module, dict[str, Any]]:
    """Load tensor/builtin-only checkpoints; never unpickle arbitrary objects."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported Holospex checkpoint format.")
    architecture = checkpoint.get("architecture")
    model_id_for_architecture(architecture)
    classes = checkpoint.get("classes")
    if not isinstance(classes, list) or len(classes) < 2 or [entry.get("index") for entry in classes] != list(range(len(classes))):
        raise ValueError("Checkpoint classes must be in contiguous training-index order.")
    size = checkpoint.get("input_size", {})
    if any(not isinstance(size.get(key), int) or isinstance(size[key], bool) or size[key] <= 0 for key in ("width", "height")):
        raise ValueError("Checkpoint must contain a positive integer input size.")
    model = build_model(num_classes=len(classes), pretrained=False, architecture=architecture)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(resolve_device(device)).eval(), checkpoint
