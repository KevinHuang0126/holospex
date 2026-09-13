"""Visual diagnostics of original-resolution masks, separate from the HUD.

The raw model argmax is shown without confidence filtering or polygon conversion.
Dataset annotations are supplied labels, not a new clinical review of the run.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .adapters import FrameInput
from .dataset import encode_mask, read_image, read_semantic_mask
from .inference import SegmentationAdapter

IGNORED_COLOR = "#92999e"
IGNORE_INDEX = 255

def _select_samples(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Spread across videos deterministically, without selecting by model quality."""
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for sample in sorted(samples, key=lambda item: (str(item["videoId"]), item["frameNumber"])):
        groups.setdefault(str(sample["videoId"]), []).append(sample)
    videos = list(groups.values())
    positions = np.linspace(0, len(videos) - 1, min(limit, len(videos)), dtype=int)
    selected = [videos[position][len(videos[position]) // 2] for position in positions]
    if len(selected) < limit:
        selected_paths = {item["imagePath"] for item in selected}
        remaining = [item for group in videos for item in group if item["imagePath"] not in selected_paths]
        selected.extend(remaining[:limit - len(selected)])
    return selected


def _thumbnail(image: np.ndarray, labels: np.ndarray | None, classes: list[dict[str, Any]], palette: dict[str, Any], *, ignore_index: int | None = None) -> Image.Image:
    """Bilinear RGB plus nearest-neighbor categorical labels; never contours."""
    base = Image.fromarray(image).resize((320, 180), Image.Resampling.BILINEAR)
    if labels is None:
        return base
    if labels.shape != image.shape[:2]:
        raise ValueError("Diagnostic mask and source image dimensions differ.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Diagnostic masks must contain integer training indices.")
    scored = labels if ignore_index is None else labels[labels != ignore_index]
    if scored.size and (scored.min() < 0 or scored.max() >= len(classes)):
        raise ValueError("Diagnostic mask contains an unknown training index.")
    resized = np.array(Image.fromarray(labels.astype(np.int32)).resize((320, 180), Image.Resampling.NEAREST))
    pixels = np.array(base, dtype=np.float32)
    for entry in classes:
        if entry["structureId"] == "background":
            continue
        color = np.array(ImageColor.getrgb(palette[entry["structureId"]]["color"]), dtype=np.float32)
        region = resized == entry["index"]
        pixels[region] = 0.58 * pixels[region] + 0.42 * color
    if ignore_index is not None:
        # Explicit unknown/uncertain annotation pixels remain visible as gray,
        # never blended into an anatomical class or counted as background.
        pixels[resized == ignore_index] = ImageColor.getrgb(IGNORED_COLOR)
    return Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8))


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        proposed = f"{current} {word}".strip()
        if current and draw.textlength(proposed, font=font) > width:
            lines.append(current)
            current = word
        else:
            current = proposed
    if current:
        lines.append(current)
    return lines


def render_comparison(
    checkpoint_path: Path, manifest: dict, output: Path, *, split: str = "val",
    limit: int = 3, device: str = "auto",
) -> Path:
    """Write a PNG contact sheet and a same-basename JSON audit record.

    Image columns are original / supplied annotation / raw model prediction.
    Frame selection is deterministic and makes no use of model confidence or
    prediction quality. The companion JSON separately describes filtered HUD
    results so an empty HUD cannot be mistaken for an empty raw prediction.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test.")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer.")
    output, checkpoint_path = Path(output), Path(checkpoint_path)
    if output.suffix.lower() != ".png":
        raise ValueError("Comparison output must use a .png extension.")
    sidecar_path = output.with_suffix(".json")
    for path in (output, sidecar_path):
        if path.exists():
            raise FileExistsError(f"Refusing to replace {path}")
    samples = [sample for sample in manifest["samples"] if sample["split"] == split]
    if not samples:
        raise ValueError(f"The manifest has no {split} samples.")
    selected = _select_samples(samples, limit)
    palette_path = Path(__file__).resolve().parents[3] / "contracts/anatomy.json"
    palette = json.loads(palette_path.read_text(encoding="utf-8"))
    classes = manifest["classes"]
    if [entry["index"] for entry in classes] != list(range(len(classes))):
        raise ValueError("Manifest classes must use contiguous ordered training indices.")
    for entry in classes:
        if entry["structureId"] != "background" and entry["structureId"] not in palette:
            raise ValueError(f"No shared anatomy color for {entry['structureId']!r}.")
    adapter = SegmentationAdapter(checkpoint_path, device=device)
    checkpoint = adapter.checkpoint
    if checkpoint["classes"] != classes:
        raise ValueError("Checkpoint and manifest class maps differ.")
    ignore_source_ids = sorted(manifest.get("ignoreSourceIds", []))
    if manifest.get("ignoreIndex", IGNORE_INDEX) != IGNORE_INDEX or checkpoint.get("ignore_index", IGNORE_INDEX) != IGNORE_INDEX:
        raise ValueError("Diagnostics require ignored targets to use index 255.")
    if sorted(checkpoint.get("ignore_source_ids", [])) != ignore_source_ids:
        raise ValueError("Checkpoint and manifest ignored-label policies differ.")

    width, header_height, row_height = 1040, 154, 222
    legend_count = len(classes) + (1 if ignore_source_ids else 0)
    legend_rows = (legend_count + 2) // 3
    height = header_height + len(selected) * row_height + 47 + legend_rows * 25
    sheet = Image.new("RGB", (width, height), "#f4f7f6")
    draw = ImageDraw.Draw(sheet)
    title_font, body_font = ImageFont.load_default(size=25), ImageFont.load_default(size=14)
    label_font, small_font = ImageFont.load_default(size=15), ImageFont.load_default(size=12)
    draw.text((24, 20), "Holospex | Segmentation comparison", fill="#14343c", font=title_font)
    checkpoint_label = f"Checkpoint: {checkpoint_path.name} | {checkpoint['model_id']} | {checkpoint['model_version']}"
    draw.text((24, 58), checkpoint_label, fill="#45616a", font=small_font)
    note = "Supplied dataset annotations and raw model argmax. No clinical review. Prediction overlays are not confidence-thresholded."
    for index, line in enumerate(_wrap(draw, note, body_font, width - 48)):
        draw.text((24, 82 + 19 * index), line, fill="#58717a", font=body_font)
    column_x = (24, 360, 696)
    for x, title in zip(column_x, ("Original frame", "Supplied annotation", "Model raw argmax")):
        draw.text((x, 127), title, fill="#14343c", font=label_font)

    frame_summaries = []
    for row_index, sample in enumerate(selected):
        image_path, mask_path = Path(sample["imagePath"]), Path(sample["maskPath"])
        image = read_image(image_path)
        truth = encode_mask(read_semantic_mask(mask_path), classes, ignore_source_ids=ignore_source_ids)
        source_height, source_width = image.shape[:2]
        if truth.shape != (source_height, source_width):
            raise ValueError(f"Image and annotation dimensions differ for {image_path}.")
        frame = FrameInput(
            media_id=f"endoscapes-video-{sample['videoId']}",
            frame_number=sample["frameNumber"], timestamp_ms=sample["timestampMs"],
            width=source_width, height=source_height, image_path=image_path,
        )
        result, raw_labels, withheld = adapter.predict_details(frame)
        y = header_height + row_index * row_height
        draw.text((24, y), f"{split} | video {sample['videoId']} | frame {sample['frameNumber']} | {sample['timestampMs']:g} ms", fill="#45616a", font=small_font)
        for x, (labels, ignored) in zip(column_x, ((None, None), (truth, IGNORE_INDEX), (raw_labels, None))):
            sheet.paste(_thumbnail(image, labels, classes, palette, ignore_index=ignored), (x, y + 24))
        frame_summaries.append({
            "videoId": sample["videoId"], "frameNumber": sample["frameNumber"],
            "timestampMs": sample["timestampMs"], "imagePath": str(image_path.resolve()),
            "annotationPath": str(mask_path.resolve()), "originalSize": {"width": source_width, "height": source_height},
            "rawPredictionPixels": {entry["structureId"]: int(np.count_nonzero(raw_labels == entry["index"])) for entry in classes},
            "annotationPixels": {entry["structureId"]: int(np.count_nonzero(truth == entry["index"])) for entry in classes},
            "ignoredAnnotationPixels": int(np.count_nonzero(truth == IGNORE_INDEX)),
            "scoredAnnotationPixels": int(np.count_nonzero(truth != IGNORE_INDEX)),
            "filteredHudResult": {
                "mediaId": result["mediaId"], "status": result["status"], "source": result["source"],
                "model": result["model"], "structureCount": len(result["structures"]),
                "structures": [{"structureId": structure["structureId"], "confidence": structure.get("confidence"), "polygonVertices": len(structure["polygon"])} for structure in result["structures"]],
            },
            "withheldHudComponents": withheld,
        })
        print(f"comparison frame={row_index + 1}/{len(selected)} video={sample['videoId']} frame_number={sample['frameNumber']}", flush=True)

    legend_y = header_height + len(selected) * row_height + 7
    draw.line((24, legend_y - 5, width - 24, legend_y - 5), fill="#d2e0db", width=1)
    draw.text((24, legend_y + 3), "Shared anatomy colors | original pixels, categorical masks resized with nearest neighbor", fill="#58717a", font=small_font)
    for index, entry in enumerate(classes):
        x, y = column_x[index % 3], legend_y + 30 + (index // 3) * 25
        background = entry["structureId"] == "background"
        label = "Background (not tinted)" if background else palette[entry["structureId"]]["label"]
        color = "#f4f7f6" if background else palette[entry["structureId"]]["color"]
        draw.rectangle((x, y, x + 11, y + 11), fill=color, outline="#8ea09a")
        draw.text((x + 19, y - 1), label, fill="#45616a", font=small_font)
    if ignore_source_ids:
        index = len(classes)
        x, y = column_x[index % 3], legend_y + 30 + (index // 3) * 25
        draw.rectangle((x, y, x + 11, y + 11), fill=IGNORED_COLOR, outline="#8ea09a")
        draw.text((x + 19, y - 1), "Ignored annotation (unscored)", fill="#45616a", font=small_font)

    audit = {
        "formatVersion": 1, "artifactType": "segmentation_comparison", "split": split,
        "checkpointPath": str(checkpoint_path.resolve()), "modelId": checkpoint["model_id"],
        "modelVersion": checkpoint["model_version"], "epochsTrained": checkpoint.get("epochs_trained"),
        "training": checkpoint.get("training", {}), "requestedLimit": limit,
        "selectedFrameCount": len(selected), "availableSplitFrames": len(samples),
        "selectionMethod": "Evenly spaced video groups; middle annotated frame in each selected video; unused frames fill any remaining requested rows. No model-quality selection.",
        "display": {"maskSource": "raw_argmax", "confidenceThresholdApplied": False, "overlayOpacity": 0.42, "thumbnailSize": {"width": 320, "height": 180}, "maskInterpolation": "nearest", "clinicalReview": False},
        "filteredHud": {"confidenceThreshold": adapter.threshold, "minimumContourAreaPixels": adapter.min_area, "confidenceDefinition": "Mean uncalibrated class softmax over accepted component pixels.", "note": "HUD summaries use thresholds and polygon filters; these filters are not applied to the rendered raw prediction column."},
        "paletteSource": str(palette_path), "classes": classes, "frames": frame_summaries,
        "ignoreIndex": IGNORE_INDEX, "ignoreSourceIds": ignore_source_ids,
        "ignoredAnnotationColor": IGNORED_COLOR,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")
    sidecar_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output.resolve()
