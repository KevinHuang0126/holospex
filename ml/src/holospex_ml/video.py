"""Offline frame-by-frame video inference using decoded presentation timestamps.

No FPS guesses, temporal propagation, or temporary JPEGs. Frame numbers refer to
zero-based decoded order in this clip, not Endoscapes source-video filenames.
Input must be a prepared video-only clip whose first presentation timestamp is
zero. A nonzero source timeline is rejected, never silently shifted for the HUD.
Run outputs belong in an ignored runs directory, not in shared demo assets.
"""

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from .adapters import FrameInput
from .validation import validate_frame_result


def _json_write(path, value, *, replace=False):
    """Publish one complete JSON file, with exclusive creation by default."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".video-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fraction_record(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def export_video(adapter, input_path: Path, output: Path, media_id: str,
                 max_frames: int | None = None) -> dict:
    """Export a zero-start video-only clip to HUD results, audit, and raw masks.

    A frame limit decodes one extra frame to distinguish a truncated prefix
    from a clip ending exactly at the limit. decodedFrameCount includes that
    lookahead; processedFrameCount and the JSON array do not. On failure, no
    final array is published; the sidecar records failure and any completed
    raw masks remain available for diagnosis. Use a fresh output path to retry.
    """
    if not isinstance(media_id, str) or not media_id.strip():
        raise ValueError("media_id must be a nonempty string")
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
        raise ValueError("max_frames must be a positive integer or None")
    input_path, output = Path(input_path).resolve(), Path(output).absolute()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output.suffix.lower() != ".json":
        raise ValueError("Video result output must have a .json extension")
    sidecar = output.with_suffix(".info.json")
    masks_dir = output.with_suffix(".masks")
    for destination in (output, sidecar, masks_dir):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to replace {destination}")
    import av  # Optional decoder dependency, loaded only for an explicit run.

    report = {
        "formatVersion": 1, "artifactType": "video_segmentation_export", "status": "running",
        "startedAt": datetime.now(timezone.utc).isoformat(), "mediaId": media_id,
        "inputPath": str(input_path), "inputSha256": _sha256(input_path),
        "inputBytes": input_path.stat().st_size, "decoder": "PyAV", "decoderVersion": av.__version__,
        "containerMetadataErrors": "replace",
        "containerMetadataPolicy": "Replace undecodable metadata text only; decoded image pixels and presentation timestamps are unchanged.",
        "outputPath": str(output), "rawMasksDirectory": masks_dir.name,
        "model": {"id": adapter.checkpoint["model_id"], "version": adapter.checkpoint["model_version"]},
        "classes": adapter.checkpoint["classes"], "training": adapter.checkpoint.get("training", {}),
        "confidenceThreshold": adapter.threshold, "minimumContourAreaPixels": adapter.min_area,
        "confidenceDefinition": "Mean uncalibrated class softmax over accepted component pixels.",
        "rawMaskDefinition": "Unthresholded per-frame argmax in original decoded pixel coordinates.",
        "frameNumberDefinition": "Zero-based decoded presentation order within this input clip.",
        "timestampDefinition": "Decoded PTS × frame time base × 1000; first presentation timestamp must be zero.",
        "timelinePolicy": "Require a prepared zero-start video-only input; do not rebase source timestamps.",
        "temporalPropagation": False, "maximumFrames": max_frames,
        "decodedFrameCount": 0, "processedFrameCount": 0, "partialLimitReached": False,
        "clipPTSBase": None, "streamTimeBase": None, "frameTimeBases": [], "frames": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir()  # Exclusive claim; never reuse a partial mask directory.
    _json_write(sidecar, report)
    results = []
    first_time = previous_time = None
    dimensions = None
    try:
        with av.open(str(input_path), metadata_errors="replace") as container:
            report["containerStartTimeUs"] = getattr(container, "start_time", None)
            report["audioStreamCount"] = len(container.streams.audio)
            if not container.streams.video:
                raise ValueError("Input contains no video stream")
            if container.streams.audio:
                raise ValueError("Video export requires a prepared video-only clip. Remove audio with ffmpeg -an before exporting.")
            stream = container.streams.video[0]
            report["videoStreamIndex"] = stream.index
            if stream.time_base is not None:
                report["streamTimeBase"] = _fraction_record(stream.time_base)
            for decoded in container.decode(stream):
                frame_number = report["decodedFrameCount"]
                report["decodedFrameCount"] += 1
                if decoded.pts is None or decoded.time_base is None:
                    raise ValueError(f"Decoded frame {frame_number} lacks a presentation timestamp/time base")
                time_base = Fraction(decoded.time_base)
                if time_base <= 0:
                    raise ValueError("Decoded frame time base must be positive")
                presented_at = decoded.pts * time_base
                if previous_time is not None and presented_at <= previous_time:
                    raise ValueError(f"Decoded frame {frame_number} has non-increasing presentation time")
                previous_time = presented_at
                current_dimensions = (decoded.width, decoded.height)
                if dimensions is None:
                    dimensions = current_dimensions
                    if min(dimensions) <= 0:
                        raise ValueError("Decoded frame dimensions must be positive")
                    report["originalSize"] = {"width": dimensions[0], "height": dimensions[1]}
                elif current_dimensions != dimensions:
                    raise ValueError("Variable frame dimensions require a separate media/renderer contract")
                if first_time is None:
                    first_time = presented_at
                    report["clipPTSBase"] = {"pts": decoded.pts, "timeBase": _fraction_record(time_base)}
                    if first_time != 0:
                        raise ValueError(
                            "First decoded presentation timestamp must be zero. Prepare a video-only clip "
                            "with ffmpeg -an and trim=...,setpts=PTS-STARTPTS, then verify the encoded clip's "
                            "first decoded PTS is zero. Source timestamps are not automatically rebased."
                        )
                base_record = _fraction_record(time_base)
                if base_record not in report["frameTimeBases"]:
                    report["frameTimeBases"].append(base_record)
                if max_frames is not None and len(results) >= max_frames:
                    report["partialLimitReached"] = True
                    break
                timestamp_ms = float(presented_at * 1000)
                if not math.isfinite(timestamp_ms):
                    raise ValueError("Frame presentation timestamp is not finite")
                rgb = decoded.to_ndarray(format="rgb24")
                frame = FrameInput(media_id=media_id, frame_number=frame_number, timestamp_ms=timestamp_ms,
                                   width=dimensions[0], height=dimensions[1])
                result, labels, withheld = adapter.predict_rgb_details(frame, rgb)
                validate_frame_result(result)
                identity = {"mediaId": media_id, "frameNumber": frame_number, "timestampMs": timestamp_ms,
                            "width": dimensions[0], "height": dimensions[1], "coordinateSpace": "original_pixels",
                            "source": "ml_prediction", "status": "ok", "model": report["model"]}
                if any(result.get(key) != value for key, value in identity.items()):
                    raise ValueError("Adapter returned a result for a different frame or without a direct prediction")
                labels = np.asarray(labels)
                if (labels.shape != (dimensions[1], dimensions[0]) or labels.dtype != np.uint8
                        or labels.max() >= len(report["classes"])):
                    raise ValueError("Adapter raw mask does not match the decoded frame/class map")
                mask_name = f"frame-{frame_number:06d}.mask.png"
                with (masks_dir / mask_name).open("xb") as handle:
                    Image.fromarray(labels).save(handle, format="PNG")
                results.append(result)
                report["processedFrameCount"] = len(results)
                report["frames"].append({"frameNumber": frame_number, "timestampMs": timestamp_ms,
                                         "pts": decoded.pts, "timeBase": base_record,
                                         "rawMask": mask_name, "withheldComponents": withheld,
                                         "structureCount": len(result["structures"])})
                if len(results) == 1 or len(results) % 25 == 0:
                    print(f"video frames processed={len(results)} timestamp_ms={timestamp_ms:g}", flush=True)
                    _json_write(sidecar, report, replace=True)
        if not results:
            raise ValueError("Video decoded no usable frames")
        report["status"] = "completed"
        report["completedAt"] = datetime.now(timezone.utc).isoformat()
        _json_write(sidecar, report, replace=True)
        _json_write(output, results)
    except BaseException as error:
        report["status"] = "failed"
        report["failedAt"] = datetime.now(timezone.utc).isoformat()
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        _json_write(sidecar, report, replace=True)
        raise
    return report
