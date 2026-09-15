"""Serve in-memory JPEG identification with one already-trained checkpoint.

Run ``python -m holospex_ml.live`` with the verified promoted current model.
For other weights, pass ``--checkpoint PATH --threshold VALUE`` explicitly.
This runner neither trains nor downloads weights, and never saves camera frames.
The browser connects through its same-origin bridge; this is not a public web
server. A non-loopback bind requires HOLOSPEX_IDENTIFICATION_TOKEN.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
from typing import Any, Callable

from . import current_model
from .adapters import FrameInput

MAX_REQUEST_BYTES = 6 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_IMAGE_PIXELS = 4 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
MAX_SAFE_INTEGER = 2**53 - 1
DATASET = "Endoscapes-Seg50"
Predictor = Callable[[FrameInput, Any, float], dict[str, Any]]


def _finite_number(value: Any, maximum: float) -> bool:
    return (type(value) in (int, float) and 0 <= value <= maximum
            and math.isfinite(value))


def _frame(value: Any) -> FrameInput:
    keys = {"mediaId", "frameNumber", "timestampMs", "width", "height"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Invalid frame metadata")
    media_id = value["mediaId"]
    if (not isinstance(media_id, str) or not 1 <= len(media_id) <= 256
            or any(ord(character) < 32 for character in media_id)):
        raise ValueError("Invalid media identity")
    if (type(value["frameNumber"]) is not int
            or not 0 <= value["frameNumber"] <= MAX_SAFE_INTEGER
            or not _finite_number(value["timestampMs"], MAX_SAFE_INTEGER)):
        raise ValueError("Invalid frame identity")
    if any(type(value[key]) is not int or not 1 <= value[key] <= MAX_IMAGE_DIMENSION
           for key in ("width", "height")):
        raise ValueError("Invalid frame dimensions")
    if value["width"] * value["height"] > MAX_IMAGE_PIXELS:
        raise ValueError("Frame dimensions exceed the pixel limit")
    return FrameInput(media_id, value["frameNumber"], value["timestampMs"],
                      value["width"], value["height"])


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON property")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Nonfinite JSON number")


def decode_request(body: bytes) -> tuple[FrameInput, Any, float | None]:
    """Validate metadata and decode an unrotated, original-size RGB JPEG in RAM."""
    from PIL import Image

    if not body or len(body) > MAX_REQUEST_BYTES:
        raise ValueError("Request size exceeds the limit")
    value = json.loads(body, object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant)
    required = {"frame", "imageBase64"}
    if (not isinstance(value, dict) or not required <= set(value)
            or set(value) - required - {"minimumConfidence"}):
        raise ValueError("Invalid identification request")
    threshold = value.get("minimumConfidence")
    if "minimumConfidence" in value and not _finite_number(threshold, 1):
        raise ValueError("Minimum confidence must be a number in [0, 1]")
    frame = _frame(value["frame"])
    encoded = value["imageBase64"]
    if not isinstance(encoded, str) or not encoded or len(encoded) % 4:
        raise ValueError("Invalid JPEG encoding")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Invalid JPEG encoding") from error
    if (base64.b64encode(data).decode("ascii") != encoded
            or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9")):
        raise ValueError("Invalid JPEG encoding")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format != "JPEG" or source.size != (frame.width, frame.height):
                raise ValueError("JPEG dimensions do not match the captured frame")
            if source.getexif().get(274, 1) != 1:
                raise ValueError("JPEG orientation must match the captured pixels")
            source.load()  # Reject truncated input before any model call.
            image = source.convert("RGB")
    except Image.DecompressionBombError as error:
        raise ValueError("JPEG dimensions exceed the pixel limit") from error
    return frame, image, threshold


def _validate_prediction(result: Any, frame: FrameInput, model: dict[str, str]) -> None:
    from .validation import validate_frame_result

    validate_frame_result(result)
    expected = {"mediaId": frame.media_id, "frameNumber": frame.frame_number,
                "timestampMs": frame.timestamp_ms, "width": frame.width,
                "height": frame.height, "source": "ml_prediction", "model": model}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Prediction does not match the capture or loaded model")


def create_server(host: str, port: int, predictor: Predictor,
                  model: dict[str, str], minimum_confidence: float,
                  token: str | None = None) -> ThreadingHTTPServer:
    """Build a runner; predictor receives (FrameInput, RGB PIL image, cutoff).

    The nonblocking lock covers reading, decoding, and inference. A stalled model
    holds that lock until it really returns; disconnected clients cannot create
    an inference queue. Camera contents and exception details are never logged.
    """
    if not _finite_number(minimum_confidence, 1):
        raise ValueError("An explicit confidence threshold in [0, 1] is required")
    if (not isinstance(model, dict) or set(model) != {"id", "version"}
            or any(not isinstance(value, str) or not 1 <= len(value) <= 256
                   for value in model.values())):
        raise ValueError("Loaded model identity is required")
    if token is not None and (not isinstance(token, str) or not token
                              or len(token) > 1024 or any(character.isspace() for character in token)):
        raise ValueError("Invalid identification token")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    if not loopback and token is None:
        raise ValueError("Non-loopback binding requires HOLOSPEX_IDENTIFICATION_TOKEN")
    loaded_model = dict(model)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _json(self, status: int, value: dict[str, Any]) -> None:
            payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
            if len(payload) > MAX_RESPONSE_BYTES:
                status = 500
                payload = b'{"status":"error","message":"Identification unavailable."}'
            self.close_connection = True
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"status": "error", "message": message})

        def _allowed(self) -> bool:
            if self.path != "/identify":
                self._error(404, "Endpoint not found.")
                return False
            if token is not None:
                values = self.headers.get_all("Authorization", [])
                expected = f"Bearer {token}".encode("utf-8")
                if len(values) != 1 or not hmac.compare_digest(values[0].encode("utf-8"), expected):
                    self._error(401, "Identification authorization required.")
                    return False
            return True

        def do_GET(self) -> None:
            if self._allowed():
                self._json(200, {"status": "ready", "model": loaded_model,
                                 "minimumConfidence": minimum_confidence,
                                 "supportsMinimumConfidence": True, "dataset": DATASET})

        def do_POST(self) -> None:
            if not self._allowed():
                return
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get_all("Transfer-Encoding") or len(lengths) != 1:
                self._error(400, "A bounded Content-Length is required.")
                return
            if not lengths[0].isascii() or not lengths[0].isdigit():
                self._error(400, "Invalid request length.")
                return
            if len(lengths[0]) > 10 or not 1 <= int(lengths[0]) <= MAX_REQUEST_BYTES:
                self._error(413, "Identification request is too large.")
                return
            content_types = self.headers.get_all("Content-Type", [])
            if len(content_types) != 1 or content_types[0].split(";", 1)[0].strip().lower() != "application/json":
                self._error(415, "Identification requires application/json.")
                return
            if not lock.acquire(blocking=False):
                self._error(429, "Identification is busy. Retry with a new frame.")
                return
            try:
                try:
                    body = self.rfile.read(int(lengths[0]))
                    if len(body) != int(lengths[0]):
                        raise ValueError("Truncated request")
                    frame, image, requested_threshold = decode_request(body)
                except (ValueError, OSError, RecursionError):
                    self._error(400, "Invalid JPEG or captured frame metadata.")
                    return
                try:
                    with image:
                        threshold = minimum_confidence if requested_threshold is None else requested_threshold
                        result = predictor(frame, image, threshold)
                    _validate_prediction(result, frame, loaded_model)
                    self._json(200, result)
                except Exception:
                    self._error(500, "Identification unavailable.")
            finally:
                lock.release()

        def _unsupported(self) -> None:
            self._error(405, "Use GET or POST for identification.")

        do_OPTIONS = do_PUT = do_PATCH = do_DELETE = do_HEAD = _unsupported

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

        def handle_error(self, request: Any, client_address: Any) -> None:
            # Socket interruption should not print request contents or tracebacks.
            pass

    return Server((host, port), Handler)


def resolve_configuration(checkpoint: Path | None, threshold: float | None) -> tuple[Path, float]:
    """Default only the promoted selection; custom weights need their own threshold."""
    checkpoint = (checkpoint or current_model.CHECKPOINT_PATH).resolve()
    if threshold is None and checkpoint == current_model.CHECKPOINT_PATH.resolve():
        threshold = current_model.MINIMUM_CONFIDENCE
    if not _finite_number(threshold, 1):
        raise ValueError("Custom checkpoints require an explicit confidence threshold in [0, 1]")
    return checkpoint, threshold


def load_predictor(checkpoint: Path | None = None, device: str = "auto",
                   threshold: float | None = None) -> tuple[Predictor, dict[str, str]]:
    """Load one trained checkpoint, verifying the current selection before Torch.

    Custom checkpoint paths retain explicit threshold and metadata requirements.
    The existing adapter supports both MobileNet and ResNet without downloading.
    """
    checkpoint, threshold = resolve_configuration(checkpoint, threshold)
    if not checkpoint.is_file():
        raise ValueError("The trained checkpoint is unavailable")
    is_current = checkpoint == current_model.CHECKPOINT_PATH.resolve()
    if is_current:
        current_model.verify_checkpoint(checkpoint)
    import numpy as np
    from .dataset import STRUCTURE_ORDER
    from .inference import SegmentationAdapter

    adapter = SegmentationAdapter(checkpoint, device=device, threshold=threshold)
    metadata = adapter.checkpoint
    dataset = metadata.get("dataset")
    dataset_name = dataset.get("dataset") if isinstance(dataset, dict) else None
    # Each reviewed training batch appends this exact suffix to its inherited
    # base dataset. Batch 002 extends the already-reviewed pilot; a base-only
    # comparison would reject the promoted checkpoint before serving any frame.
    if (not isinstance(dataset_name, str)
            or re.fullmatch(r"endoscapes-seg50(?:\+reviewed-partial)*", dataset_name) is None
            or tuple(entry.get("structureId") for entry in metadata["classes"]) != STRUCTURE_ORDER):
        raise ValueError("Checkpoint must be the Endoscapes-Seg50 anatomy model")
    if is_current:
        current_model.validate_metadata(metadata)
        current_model.verify_checkpoint(checkpoint)  # Refuse a replacement during loading.
    model = {"id": metadata["model_id"], "version": metadata["model_version"]}

    def predict(frame: FrameInput, image: Any, minimum_confidence: float | None = None) -> dict[str, Any]:
        return adapter.predict_rgb_details(frame, np.asarray(image), threshold=minimum_confidence)[0]

    return predict, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        help="Existing trained weights; default: verified ml/weights/current/best.pt")
    parser.add_argument("--threshold", type=float,
                        help="Minimum softmax confidence in [0, 1]; current model defaults to 0.5; required for other weights")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:INDEX")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address; non-loopback requires HOLOSPEX_IDENTIFICATION_TOKEN")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("Invalid port")
        checkpoint, threshold = resolve_configuration(args.checkpoint, args.threshold)
        predictor, model = load_predictor(checkpoint, args.device, threshold)
        server = create_server(args.host, args.port, predictor, model, threshold,
                               os.environ.get("HOLOSPEX_IDENTIFICATION_TOKEN"))
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError):
        print("Identification runner unavailable. Check ml/CURRENT_MODEL.md, the trained checkpoint, ML dependencies, "
              "threshold, device, and binding/token configuration.", file=sys.stderr)
        return 1
    print(f"Identification runner ready on port {args.port}: {model['id']} {model['version']} "
          f"(threshold {threshold}); camera frames are not saved.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
