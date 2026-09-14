"""Annotation-selected small-anatomy crops; inference always sees whole frames.

These are diagnostic examples, not a model-selection metric or clinical review.
The original-grid, whole-frame scores in the sidecar include errors outside the
display crop. Raw argmax masks bypass HUD confidence and contour filtering.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .adapters import FrameInput
from .dataset import encode_mask, read_image, read_semantic_mask
from .diagnostics import IGNORED_COLOR
from .inference import SegmentationAdapter
from .metrics import IGNORE_INDEX, confusion_matrix, metrics_from_confusion

FOCUS_CLASSES = (
    "cystic_duct", "cystic_artery", "cystic_plate",
    "hepatocystic_triangle_dissection",
)
CONTEXT_PIXELS = 64
SELECTION_RULE = (
    "For each focus class, select the upper-median original annotation pixel "
    "area among validation frames containing that class. Sort by area, then "
    "string video ID, frame number, and image path. No prediction, confidence, "
    "or model-quality selection; the same frame may serve multiple rows."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_focus_samples(samples, classes, ignore_source_ids):
    """Inspect validation annotations only; do not load either model here."""
    indices = {entry["structureId"]: entry["index"] for entry in classes}
    if any(focus not in indices for focus in FOCUS_CLASSES):
        raise ValueError("Manifest must include all four small-anatomy focus classes.")
    candidates = {focus: [] for focus in FOCUS_CLASSES}
    for sample in samples:
        if sample["split"] != "val":
            continue
        truth = encode_mask(read_semantic_mask(Path(sample["maskPath"])), classes, ignore_source_ids)
        for focus in FOCUS_CLASSES:
            area = int(np.count_nonzero(truth == indices[focus]))
            if area:
                key = (area, str(sample["videoId"]), sample["frameNumber"], str(sample["imagePath"]))
                candidates[focus].append((key, sample))
    selected = []
    for focus in FOCUS_CLASSES:
        ordered = sorted(candidates[focus], key=lambda item: item[0])
        if not ordered:
            raise ValueError(f"No validation annotation contains {focus}.")
        rank = len(ordered) // 2
        key, sample = ordered[rank]
        selected.append((focus, sample, key[0], len(ordered), rank))
    return selected


def _crop_box(truth: np.ndarray, focus_index: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(truth == focus_index)
    if not len(xs):
        raise ValueError("Selected focus class is absent from its annotation.")
    height, width = truth.shape
    return (max(0, int(xs.min()) - CONTEXT_PIXELS),
            max(0, int(ys.min()) - CONTEXT_PIXELS),
            min(width, int(xs.max()) + 1 + CONTEXT_PIXELS),
            min(height, int(ys.max()) + 1 + CONTEXT_PIXELS))


def _crop_view(image, labels, crop, classes, palette, *, annotation=False):
    """Letterbox a display crop without changing the inference or score grid."""
    left, top, right, bottom = crop
    rgb = image[top:bottom, left:right]
    width, height = right - left, bottom - top
    scale = min(320 / width, 180 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    pixels = np.array(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    if labels is not None:
        categorical = np.array(Image.fromarray(labels[top:bottom, left:right].astype(np.int32)).resize(size, Image.Resampling.NEAREST))
        for entry in classes:
            if entry["structureId"] == "background":
                continue
            region = categorical == entry["index"]
            color = np.array(ImageColor.getrgb(palette[entry["structureId"]]["color"]))
            pixels[region] = 0.58 * pixels[region] + 0.42 * color
        if annotation:
            pixels[categorical == IGNORE_INDEX] = ImageColor.getrgb(IGNORED_COLOR)
    canvas = Image.new("RGB", (320, 180), "#dfe6e3")
    canvas.paste(Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8)),
                 ((320 - size[0]) // 2, (180 - size[1]) // 2))
    return canvas


def _checkpoint_details(adapter, path, classes, ignore_source_ids, validation_videos):
    checkpoint = adapter.checkpoint
    if checkpoint["classes"] != classes:
        raise ValueError("Checkpoint and manifest class maps differ.")
    if (checkpoint.get("ignore_index", IGNORE_INDEX) != IGNORE_INDEX
            or sorted(checkpoint.get("ignore_source_ids", [])) != ignore_source_ids):
        raise ValueError("Checkpoint and manifest ignored-label policies differ.")
    train_videos = checkpoint.get("training", {}).get("train_video_ids")
    if not isinstance(train_videos, list):
        raise ValueError("Checkpoint lacks training video IDs; cannot verify validation separation.")
    if validation_videos & {str(video) for video in train_videos}:
        raise ValueError("Validation examples overlap checkpoint training videos.")
    return {
        "path": str(path.resolve()), "sha256": _sha256(path),
        "modelId": checkpoint["model_id"], "modelVersion": checkpoint["model_version"],
        "epochsTrained": checkpoint.get("epochs_trained"),
        "inputSize": checkpoint["input_size"],
    }


def render_small_anatomy_comparison(
    before_checkpoint: Path, after_checkpoint: Path, manifest: dict,
    output: Path, device: str = "auto",
) -> Path:
    """Save a four-column PNG plus same-stem JSON; refuse existing outputs."""
    output = Path(output)
    sidecar_path = output.with_suffix(".json")
    if output.suffix.lower() != ".png":
        raise ValueError("Comparison output must use a .png extension.")
    for path in (output, sidecar_path):
        if path.exists():
            raise FileExistsError(f"Refusing to replace {path}")
    classes = manifest["classes"]
    if (not classes or classes[0].get("structureId") != "background"
            or [entry["index"] for entry in classes] != list(range(len(classes)))):
        raise ValueError("Manifest needs ordered contiguous classes beginning with background.")
    if manifest.get("ignoreIndex", IGNORE_INDEX) != IGNORE_INDEX:
        raise ValueError("Comparison requires ignored targets to use index 255.")
    ignore_source_ids = sorted(manifest.get("ignoreSourceIds", []))
    validation_samples = [sample for sample in manifest["samples"] if sample["split"] == "val"]
    selected = _select_focus_samples(validation_samples, classes, ignore_source_ids)
    validation_videos = {str(sample["videoId"]) for sample in validation_samples}
    palette_path = Path(__file__).resolve().parents[3] / "contracts/anatomy.json"
    palette = json.loads(palette_path.read_text(encoding="utf-8"))
    for entry in classes[1:]:
        if entry["structureId"] not in palette:
            raise ValueError(f"No shared anatomy color for {entry['structureId']}.")
    paths = [Path(before_checkpoint), Path(after_checkpoint)]
    adapters = [SegmentationAdapter(path, device=device) for path in paths]
    checkpoints = [_checkpoint_details(adapter, path, classes, ignore_source_ids, validation_videos)
                   for adapter, path in zip(adapters, paths)]

    width, header_height, row_height = 1376, 194, 230
    height = header_height + len(selected) * row_height + 119
    sheet = Image.new("RGB", (width, height), "#f4f7f6")
    draw = ImageDraw.Draw(sheet)
    title_font, font = ImageFont.load_default(size=24), ImageFont.load_default(size=14)
    small_font = ImageFont.load_default(size=12)
    draw.text((24, 18), "Holospex | Small-anatomy before / after", fill="#14343c", font=title_font)
    notes = (
        "Supplied dataset annotations; no clinical review. Examples selected by median annotation area, without model-quality selection.",
        "Whole-frame inference; annotation crop + 64px context for display only. All class colors shown; raw argmax has no HUD filtering.",
        "Each row reports the focus class IoU over the WHOLE original frame, including errors outside the crop; annotation ID 255 is unscored.",
    )
    for row, note in enumerate(notes):
        draw.text((24, 55 + row * 21), note, fill="#45616a", font=small_font)
    for row, (label, details) in enumerate(zip(("Before", "After"), checkpoints)):
        size = details["inputSize"]
        note = f"{label}: {Path(details['path']).parent.name}/{Path(details['path']).name} | epoch {details['epochsTrained']} | input {size['width']}x{size['height']} | {details['modelVersion']}"
        draw.text((24, 123 + row * 18), note, fill="#45616a", font=small_font)
    column_x = (24, 360, 696, 1032)
    for x, title in zip(column_x, ("Source crop", "Supplied annotation", "Before: raw argmax", "After: raw argmax")):
        draw.text((x, 169), title, fill="#14343c", font=font)

    cache, rows = {}, []
    for row_index, (focus, sample, area, candidate_count, median_rank) in enumerate(selected):
        image_path, mask_path = Path(sample["imagePath"]), Path(sample["maskPath"])
        key = (str(image_path.resolve()), str(mask_path.resolve()), str(sample["videoId"]), sample["frameNumber"])
        if key not in cache:
            image = read_image(image_path)
            truth = encode_mask(read_semantic_mask(mask_path), classes, ignore_source_ids)
            source_height, source_width = image.shape[:2]
            if truth.shape != (source_height, source_width):
                raise ValueError(f"Image and annotation dimensions differ for {image_path}.")
            frame = FrameInput(media_id=f"endoscapes-video-{sample['videoId']}",
                               frame_number=sample["frameNumber"], timestamp_ms=sample["timestampMs"],
                               width=source_width, height=source_height, image_path=image_path)
            predictions, scores = [], []
            for adapter in adapters:
                # Never pass the annotation crop to a model. Adapter restores
                # logits to the original frame grid before computing argmax.
                _, labels, _ = adapter.predict_rgb_details(frame, image)
                labels = np.asarray(labels)
                if (labels.shape != truth.shape or not np.issubdtype(labels.dtype, np.integer)
                        or labels.min() < 0 or labels.max() >= len(classes)):
                    raise ValueError("Model raw argmax must contain valid original-grid training indices.")
                confusion = confusion_matrix(truth, labels, len(classes))
                scores.append(metrics_from_confusion(confusion, classes,
                    ignored_pixels=int(np.count_nonzero(truth == IGNORE_INDEX))))
                predictions.append(labels)
            cache[key] = (image, truth, predictions, scores)
        image, truth, predictions, scores = cache[key]
        focus_index = next(entry["index"] for entry in classes if entry["structureId"] == focus)
        crop = _crop_box(truth, focus_index)
        focus_scores = [score["per_class"][focus_index]["iou"] for score in scores]
        y = header_height + row_index * row_height
        title = f"{palette[focus]['label']} | val video {sample['videoId']}, frame {sample['frameNumber']} | full-frame focus IoU: {focus_scores[0]:.3f} -> {focus_scores[1]:.3f}"
        draw.text((24, y), title, fill="#14343c", font=font)
        for x, labels, annotation in zip(column_x, (None, truth, *predictions), (False, True, False, False)):
            sheet.paste(_crop_view(image, labels, crop, classes, palette, annotation=annotation), (x, y + 25))
        rows.append({
            "focusClass": focus, "annotationAreaPixels": area,
            "selectionCandidateCount": candidate_count, "selectedSortedRankZeroBased": median_rank,
            "videoId": sample["videoId"], "frameNumber": sample["frameNumber"], "timestampMs": sample["timestampMs"],
            "imagePath": str(image_path.resolve()), "imageSha256": _sha256(image_path),
            "annotationPath": str(mask_path.resolve()), "annotationSha256": _sha256(mask_path),
            "originalSize": {"width": image.shape[1], "height": image.shape[0]},
            "cropBox": dict(zip(("left", "top", "rightExclusive", "bottomExclusive"), crop)),
            "wholeFrameOriginalGridMetrics": {"before": scores[0], "after": scores[1]},
        })
        print(f"small-anatomy comparison row={row_index + 1}/{len(selected)} class={focus} video={sample['videoId']} frame={sample['frameNumber']}", flush=True)

    legend_y = header_height + len(selected) * row_height
    draw.text((24, legend_y), "Shared class colors | background is not tinted | crops are letterboxed, labels resized with nearest neighbor", fill="#45616a", font=small_font)
    legend = [("Background (not tinted)", "#f4f7f6")]
    legend += [(palette[entry["structureId"]]["label"], palette[entry["structureId"]]["color"]) for entry in classes[1:]]
    legend.append(("Ignored annotation (unscored)", IGNORED_COLOR))
    for index, (label, color) in enumerate(legend):
        x, y = column_x[index % 4], legend_y + 26 + index // 4 * 28
        draw.rectangle((x, y, x + 11, y + 11), fill=color, outline="#8ea09a")
        draw.text((x + 18, y - 1), label, fill="#45616a", font=small_font)
    audit = {
        "formatVersion": 1, "artifactType": "small_anatomy_before_after", "split": "val",
        "selectionRule": SELECTION_RULE, "availableValidationFrames": len(validation_samples),
        "manifestSha256": hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "checkpoints": {"before": checkpoints[0], "after": checkpoints[1]},
        "inferenceScope": "whole original frame, independently resized to each checkpoint input size; logits restored to original size before argmax",
        "metricScope": "whole original frame, never cropped; each class IoU includes all scored pixels outside the displayed crop",
        "display": {"cropContextPixels": CONTEXT_PIXELS, "cropCoordinateSpace": "original_pixels",
                    "maskSource": "raw_argmax", "confidenceThresholdApplied": False, "contourFilteringApplied": False,
                    "overlayOpacity": 0.42, "thumbnailSize": {"width": 320, "height": 180},
                    "aspectRatio": "preserved with letterboxing", "maskInterpolation": "nearest", "clinicalReview": False},
        "ignoreIndex": IGNORE_INDEX, "ignoreSourceIds": ignore_source_ids,
        "ignoredPolicy": "Ignored annotation pixels are gray in ground truth and excluded from all metrics; never treated as background.",
        "paletteSource": str(palette_path), "paletteSha256": _sha256(palette_path),
        "classes": classes, "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output.open("xb") as stream:
            created = True
            sheet.save(stream, format="PNG")
        with sidecar_path.open("x", encoding="utf-8") as stream:
            json.dump(audit, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except Exception:
        if created:
            output.unlink(missing_ok=True)
        raise
    return output.resolve()
