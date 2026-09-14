"""The ML lead's promoted model from CURRENT_MODEL.md, without loading Torch.

Git carries this pin, not weights. Installing a checkpoint is a separate,
explicit operation; neither importing this module nor starting live inference
downloads data or model files.
"""

import hashlib
from pathlib import Path
from typing import Any

CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "weights/current/best.pt"
CHECKPOINT_SHA256 = "b406ed42ab0394edba22e1dde0edc2865a6346adb61c4bea7ab0bc00d08e1911"
CHECKPOINT_BYTES = 168351963
MODEL_ID = "holospex-deeplabv3-resnet50"
MODEL_VERSION = "2026-09-14T04:29:29.757933Z-epoch-34"
ARCHITECTURE = "deeplabv3_resnet50"
INPUT_SIZE = {"width": 672, "height": 384}
NORMALIZATION = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
MINIMUM_CONFIDENCE = 0.5


def verify_checkpoint(path: Path) -> None:
    """Reject missing, incomplete or different weights before loading tensors."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("Current model weights are missing; see ml/CURRENT_MODEL.md")
    with path.open("rb") as stream:
        if path.stat().st_size != CHECKPOINT_BYTES:
            raise ValueError("Checkpoint size does not match the promoted current model")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint checksum does not match the promoted current model")


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Bind advertised identity and adapter preprocessing to the promoted model.

    The existing SegmentationAdapter reads input_size and normalization directly,
    uses bilinear RGB resizing, and restores logits to original capture dimensions
    before extracting geometry. Do not replace that preprocessing in the runner.
    """
    expected = {
        "model_id": MODEL_ID, "model_version": MODEL_VERSION,
        "architecture": ARCHITECTURE, "input_size": INPUT_SIZE,
        "normalization": NORMALIZATION,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Loaded model identity or preprocessing differs from the current model selection")
