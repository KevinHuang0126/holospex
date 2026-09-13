"""Replace the unconfigured adapter when a suitable model is available.

Keep model-specific dependencies, class mappings, thresholds, preprocessing,
and device selection inside an adapter. The browser consumes the shared frame
contract, never model tensors. A surgical-video adapter is independent of the
physical model's marker tracking; training one does not implement the other.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

FrameResult = dict[str, Any]


@dataclass(frozen=True)
class FrameInput:
    """Metadata for a decoded frame in the original media timeline.

    A real decoder must verify width/height against its image and preserve the
    media timestamp (not wall-clock inference time). ``image_path=None`` is
    valid when decoded RGB is supplied separately to ``predict_rgb_details``
    or when the unconfigured adapter does not read image content.
    """

    media_id: str
    frame_number: int
    timestamp_ms: float
    width: int
    height: int
    image_path: Path | None = None


class InferenceAdapter(Protocol):
    """Return one result; validate against contracts before persisting it.

    A working model that detects nothing returns status ``ok`` with an empty
    structures array. Unsupported input, unavailable inference, and failures
    must not be disguised as successful empty detections. Never infer a lesson
    answer or clinical criterion solely from a segmentation confidence score.
    """

    def predict(self, frame: FrameInput) -> FrameResult: ...


class UnconfiguredAdapter:
    """An honest wiring stub. It performs no inference and emits no geometry."""

    def predict(self, frame: FrameInput) -> FrameResult:
        return {
            "schemaVersion": "1.0.0",
            "mediaId": frame.media_id,
            "frameNumber": frame.frame_number,
            "timestampMs": frame.timestamp_ms,
            "width": frame.width,
            "height": frame.height,
            "coordinateSpace": "original_pixels",
            "source": "ml_prediction",
            "status": "unsupported",
            "statusReason": "No inference model is configured; this scaffold produces no predictions.",
            "model": {"id": "unconfigured", "version": "0"},
            "structures": [],
        }
