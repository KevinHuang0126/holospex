"""Prepare an auditable Endoscapes-Seg50 manifest without loading a model.

The PNG's label IDs come from seg_label_map.txt, not COCO category IDs. Missing
semantic masks are unavailable labels, never background examples. Keep this
module usable before installing a training framework.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


class DatasetError(ValueError):
    """Dataset inputs are incomplete or inconsistent; preparation must stop."""


STRUCTURE_ORDER = (
    "background", "gallbladder", "cystic_duct", "cystic_artery", "cystic_plate",
    "hepatocystic_triangle_dissection", "tool",
)
_ALIASES = {
    "background": "background", "bg": "background",
    "gallbladder": "gallbladder",
    "cysticduct": "cystic_duct", "cysticartery": "cystic_artery",
    "cysticplate": "cystic_plate",
    "hepatocystictriangledissection": "hepatocystic_triangle_dissection",
    "hepatocystictriangle": "hepatocystic_triangle_dissection",
    "calottriangle": "hepatocystic_triangle_dissection",
    "tool": "tool", "tools": "tool", "surgicaltool": "tool",
}


def _structure_id(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", name.casefold())
    if normalized not in _ALIASES:
        raise DatasetError(f"Unrecognized anatomy label {name!r}; supply an explicit supported label mapping")
    return _ALIASES[normalized]


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
        raise DatasetError(f"{label} must be a nonnegative integer, got {value!r}")
    return int(value)


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise DatasetError(f"Dataset path escapes root {root}: {path}")
    return resolved


def _require_file(root: Path, path: Path) -> Path:
    resolved = _inside(root, path)
    if not resolved.is_file():
        raise DatasetError(f"Required dataset file is missing: {path}")
    return resolved


def read_image(path: Path) -> np.ndarray:
    """Read RGB pixels without implicit cropping, scaling, or EXIF rotation."""
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB")).copy()
    except (OSError, ValueError) as error:
        raise DatasetError(f"Cannot read image {path}: {error}") from error


def read_semantic_mask(path: Path) -> np.ndarray:
    """Return source IDs; accept an indexed/grayscale PNG or identical RGB IDs.

    Never convert an RGB mask to luminance: that could silently manufacture
    different class IDs. Palette PNG indices are label IDs, not display colors.
    """
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise DatasetError(f"Semantic masks must be PNG files: {path}")
            mask = np.asarray(image).copy()
    except (OSError, ValueError) as error:
        if isinstance(error, DatasetError):
            raise
        raise DatasetError(f"Cannot read semantic mask {path}: {error}") from error
    if mask.ndim == 3 and mask.shape[2] == 3:
        if not (np.array_equal(mask[:, :, 0], mask[:, :, 1]) and np.array_equal(mask[:, :, 0], mask[:, :, 2])):
            raise DatasetError(f"RGB mask channels differ in {path}; expected repeated label IDs, not a color image")
        mask = mask[:, :, 0]
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise DatasetError(f"Unsupported semantic mask shape/type in {path}: {mask.shape}, {mask.dtype}")
    if np.any(mask < 0):
        raise DatasetError(f"Negative semantic label IDs in {path}")
    return mask


def _ignored_ids(classes: Sequence[Mapping[str, Any]], ignore_source_ids: Sequence[int]) -> set[int]:
    ignored = {_integer(value, "Ignored source ID") for value in ignore_source_ids}
    known = {int(item["sourceId"]) for item in classes}
    if ignored & known:
        raise DatasetError(f"Cannot ignore known anatomy/background source IDs: {sorted(ignored & known)}")
    return ignored


def encode_mask(mask: np.ndarray, classes: Sequence[Mapping[str, Any]], ignore_source_ids: Sequence[int] = ()) -> np.ndarray:
    """Map supplied class IDs; explicitly ignored IDs become sentinel255.

    Ignoring an unmapped ID excludes it from supervision and metrics. It does
    not assign a meaning to that source value or relabel it as background.
    """
    mapping = {int(item["sourceId"]): int(item["index"]) for item in classes}
    ignored = _ignored_ids(classes, ignore_source_ids)
    observed = {int(value) for value in np.unique(mask)}
    unexpected = observed - mapping.keys() - ignored
    if unexpected:
        raise DatasetError(f"Unknown semantic label IDs {sorted(unexpected)}; check seg_label_map.txt")
    encoded = np.empty(mask.shape, dtype=np.int64)
    for source_id in observed:
        encoded[mask == source_id] = 255 if source_id in ignored else mapping[source_id]
    return encoded


def _read_label_map(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    pairs: list[tuple[Any, Any]] = []
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict):
        pairs = list(decoded.items())
    else:
        lines = [line.split("#", 1)[0].strip().rstrip(",") for line in text.splitlines()]
        lines = [line for line in lines if line]
        # The official release's seg_label_map.txt has one class name per line;
        # its zero-based line position is the semantic PNG ID. This is a mapping
        # from the supplied file, never an assumption about COCO category IDs.
        if lines and all(re.sub(r"[^a-z0-9]", "", line.casefold()) in _ALIASES for line in lines):
            pairs = list(enumerate(lines))
            lines = []
        for number, line in enumerate(lines, start=1):
            line = line.split("#", 1)[0].strip().rstrip(",")
            if not line:
                continue
            # Explicit IDs are required; a line-number assumption risks swapping
            # classes silently when consuming a different release's label map.
            match = re.fullmatch(r"[\"']?(\d+)[\"']?\s*[:=,\t ]\s*[\"']?(.+?)[\"']?", line)
            reverse = re.fullmatch(r"[\"']?(.+?)[\"']?\s*[:=,\t]\s*(\d+)", line)
            if match:
                pairs.append((match[1], match[2].strip(" \"'")))
            elif reverse:
                pairs.append((reverse[2], reverse[1].strip(" \"'")))
            else:
                raise DatasetError(f"Cannot parse {path}:{number}: {line!r}; expected an explicit ID and anatomy name")
    mapped: dict[str, int] = {}
    used: set[int] = set()
    for first, second in pairs:
        if str(first).isdigit():
            source_id, label = _integer(first, "Semantic label ID"), str(second)
        else:
            source_id, label = _integer(second, "Semantic label ID"), str(first)
        structure = _structure_id(label)
        if source_id in used or structure in mapped:
            raise DatasetError(f"Duplicate label ID or anatomy name in {path}: {first!r}: {second!r}")
        used.add(source_id)
        mapped[structure] = source_id
    if set(mapped) != set(STRUCTURE_ORDER):
        raise DatasetError(f"{path} must explicitly map background plus all six anatomy classes; missing {sorted(set(STRUCTURE_ORDER) - mapped.keys())}")
    return [{"index": index, "sourceId": mapped[structure], "structureId": structure}
            for index, structure in enumerate(STRUCTURE_ORDER)]


def _read_split(path: Path) -> set[int]:
    text = path.read_text(encoding="utf-8-sig")
    tokens = re.split(r"[\s,]+", text.strip().strip("[]"))
    values = []
    for token in tokens:
        if not token:
            continue
        try:
            # Official split files use numpy.savetxt scientific notation, e.g.
            # 1.850000000000000000e+02. Preserve integral identity exactly.
            value = Decimal(token)
        except InvalidOperation as error:
            raise DatasetError(f"Invalid video ID {token!r} in {path}") from error
        if not value.is_finite() or value < 0 or value != value.to_integral_value():
            raise DatasetError(f"Video IDs must be finite nonnegative integers in {path}: {token!r}")
        values.append(int(value))
    if not values:
        raise DatasetError(f"Empty video split file: {path}")
    if len(values) != len(set(values)):
        raise DatasetError(f"Duplicate video IDs in split file: {path}")
    return set(values)


def _frame_identity(image: Mapping[str, Any]) -> tuple[int, int]:
    stem = Path(str(image.get("file_name", ""))).stem
    match = re.fullmatch(r"(\d+)_(\d+)", stem)
    video = next((image[key] for key in ("video_id", "videoId") if image.get(key) is not None), match[1] if match else None)
    # The official release has frame_id:null. Its image ID concatenates video
    # and frame identity and must never be interpreted as a frame number.
    frame = next((image[key] for key in ("frame_number", "frame_id", "frameNumber") if image.get(key) is not None), match[2] if match else None)
    if video is None or frame is None:
        raise DatasetError(f"Cannot determine video/frame identity for {image.get('file_name')!r}; expected explicit metadata or VIDEO_FRAME filename")
    return _integer(video, "Video ID"), _integer(frame, "Frame number")


def _find_image(root: Path, split_root: Path, file_name: str) -> Path:
    relative = Path(file_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise DatasetError(f"COCO file_name must be a path inside the dataset: {file_name!r}")
    candidates = [split_root / relative, split_root / "images" / relative,
                  root / relative, root / "all" / relative, root / "images" / relative]
    found = {_inside(root, path) for path in candidates if path.is_file()}
    if len(found) != 1:
        raise DatasetError(f"Expected one image for {file_name!r}; found {len(found)} under {root}. Check the official images/all links and extracted files.")
    return found.pop()


def _find_mask(root: Path, split_root: Path, file_name: str) -> Path | None:
    relative = Path(file_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise DatasetError(f"Invalid COCO file_name: {file_name!r}")
    # The official archive stores masks in root/semseg; allow split-local
    # layouts as well, but reject conflicting duplicate files rather than
    # silently choosing a possibly stale annotation.
    candidates = [directory / name
                  for directory in (root / "semseg", split_root / "semseg")
                  for name in (relative.with_suffix(".png"), Path(f"{relative.stem}.png"))]
    found = {_inside(root, path) for path in candidates if path.is_file()}
    if len(found) > 1:
        raise DatasetError(f"Ambiguous semantic masks for {file_name!r}")
    return found.pop() if found else None


def prepare_dataset(root: Path, fps: float, ignore_source_ids: Sequence[int] = (), exclude_frames: Sequence[str] = ()) -> dict[str, Any]:
    """Validate the local Seg50 release and return absolute-path sample records.

    ``fps`` must match the source video frame-number timebase. It is recorded in
    the manifest rather than inferred from the one-frame-per-second sampling
    frequency. Verify the supplied value against the filename timebase before
    using timestamps for playback; capture fps need not equal filename fps.
    """
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise DatasetError(f"Dataset directory does not exist: {root}")
    if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise DatasetError("fps must be a finite positive source-video frame rate")
    classes = _read_label_map(_require_file(root, root / "seg_label_map.txt"))
    ignored = _ignored_ids(classes, ignore_source_ids)
    excluded_requested: set[str] = set()
    for frame in exclude_frames:
        if not isinstance(frame, str) or Path(frame).name != frame or not re.fullmatch(r"\d+_\d+(?:\.(?:png|jpg))?", frame):
            raise DatasetError(f"Excluded frame must be a VIDEO_FRAME identity or filename, got {frame!r}")
        excluded_requested.add(Path(frame).stem)
    excluded_matched: set[str] = set()
    excluded_report: list[dict[str, Any]] = []
    split_names = ("train", "val", "test")
    splits = {split: _read_split(_require_file(root, root / f"{split}_seg_vids.txt"))
              for split in split_names}
    for index, split in enumerate(split_names):
        for other in split_names[index + 1:]:
            overlap = splits[split] & splits[other]
            if overlap:
                raise DatasetError(f"Case-level split leakage between {split} and {other}: videos {sorted(overlap)}")

    samples: list[dict[str, Any]] = []
    counts = {split: 0 for split in split_names}
    skipped = {split: 0 for split in split_names}
    observed_labels: set[int] = set()
    ignored_counts = {split: {"images": 0, "pixels": 0,
                             "bySourceId": {str(value): {"images": 0, "pixels": 0} for value in sorted(ignored)}}
                      for split in split_names}
    distribution = {split: {item["structureId"]: {"images": 0, "pixels": 0}
                            for item in classes} for split in split_names}
    structure_by_source = {item["sourceId"]: item["structureId"] for item in classes}
    seen_frames: set[tuple[int, int]] = set()
    for split in split_names:
        split_root = root / f"{split}_seg"
        annotation_path = _require_file(root, split_root / "annotation_coco.json")
        try:
            coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"Cannot read COCO metadata {annotation_path}: {error}") from error
        categories: dict[int, str] = {}
        for category in coco.get("categories", []):
            category_id = _integer(category["id"], "COCO category ID")
            if category_id in categories:
                raise DatasetError(f"Duplicate COCO category ID {category_id} in {annotation_path}")
            categories[category_id] = _structure_id(category["name"])
        if not categories:
            raise DatasetError(f"No named COCO categories in {annotation_path}")
        images = coco.get("images", [])
        image_ids = [_integer(image["id"], "COCO image ID") for image in images]
        image_id_set = set(image_ids)
        if len(image_ids) != len(image_id_set):
            raise DatasetError(f"Duplicate COCO image IDs in {annotation_path}")
        annotated: set[int] = set()
        for annotation in coco.get("annotations", []):
            image_id = _integer(annotation["image_id"], "COCO annotation image ID")
            if image_id not in image_id_set:
                raise DatasetError(f"COCO annotation refers to unknown image {image_id} in {annotation_path}")
            if _integer(annotation["category_id"], "COCO category ID") not in categories:
                raise DatasetError(f"COCO annotation refers to an unknown category in {annotation_path}")
            if annotation.get("segmentation"):
                annotated.add(image_id)
        for image in images:
            file_name = image.get("file_name")
            if not isinstance(file_name, str) or not file_name:
                raise DatasetError(f"Missing COCO file_name in {annotation_path}")
            mask_path = _find_mask(root, split_root, file_name)
            frame_identity = Path(file_name).stem
            if frame_identity in excluded_requested:
                excluded_matched.add(frame_identity)
                values = sorted(int(value) for value in np.unique(read_semantic_mask(mask_path))) if mask_path else []
                unmapped = sorted(set(values) - structure_by_source.keys() - ignored)
                reason = (f"Explicit data-quality exclusion: unmapped semantic source IDs {unmapped}"
                          if unmapped else "Explicit frame exclusion requested before fitting")
                excluded_report.append({"frame": frame_identity, "split": split, "reason": reason,
                                        "observedSourceIds": values, "unmappedSourceIds": unmapped})
                continue
            if mask_path is None:
                if _integer(image["id"], "COCO image ID") in annotated:
                    raise DatasetError(f"Missing semantic mask for segmentation-annotated image {file_name!r} in {root / 'semseg'} or {split_root / 'semseg'}; re-extract the release")
                skipped[split] += 1
                continue
            video_id, frame_number = _frame_identity(image)
            if video_id not in splits[split]:
                raise DatasetError(f"Video {video_id} in {split} is absent from {split}_seg_vids.txt")
            identity = (video_id, frame_number)
            if identity in seen_frames:
                raise DatasetError(f"Duplicate video/frame sample: {identity}")
            seen_frames.add(identity)
            image_path = _find_image(root, split_root, file_name)
            rgb = read_image(image_path)
            mask = read_semantic_mask(mask_path)
            if rgb.shape[:2] != mask.shape:
                raise DatasetError(f"Image/mask dimension mismatch for {file_name}: {rgb.shape[:2]} versus {mask.shape}")
            height, width = mask.shape
            if _integer(image.get("width"), "COCO image width") != width or _integer(image.get("height"), "COCO image height") != height:
                raise DatasetError(f"COCO dimensions disagree with decoded image for {file_name!r}")
            encode_mask(mask, classes, ignore_source_ids=tuple(ignored))
            values, pixel_counts = np.unique(mask, return_counts=True)
            observed_labels.update(int(value) for value in values)
            if any(int(value) in ignored for value in values):
                ignored_counts[split]["images"] += 1
            for value, pixel_count in zip(values, pixel_counts):
                if int(value) in ignored:
                    ignored_counts[split]["pixels"] += int(pixel_count)
                    ignored_counts[split]["bySourceId"][str(int(value))]["images"] += 1
                    ignored_counts[split]["bySourceId"][str(int(value))]["pixels"] += int(pixel_count)
                    continue
                count = distribution[split][structure_by_source[int(value)]]
                count["images"] += 1
                count["pixels"] += int(pixel_count)
            samples.append({"split": split, "imagePath": str(image_path), "maskPath": str(mask_path),
                            "videoId": video_id, "frameNumber": frame_number,
                            "timestampMs": frame_number * 1000 / fps})
            counts[split] += 1
        if counts[split] == 0:
            raise DatasetError(f"No available semantic masks in {split_root}; verify dataset extraction")

    if excluded_requested - excluded_matched:
        raise DatasetError(f"Requested excluded frames were not found in COCO metadata: {sorted(excluded_requested - excluded_matched)}")

    source_record = None
    for path in sorted(root.glob("download-*.json"), reverse=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("subset") == "Endoscapes-Seg50" and not record.get("metadataOnly"):
            source_record = {key: record.get(key) for key in
                             ("sourceUrl", "archiveEtag", "selectedEntriesSha256", "downloadedAt")}
            break
    return {
        "schemaVersion": "1.0.0", "dataset": "endoscapes-seg50", "root": str(root),
        "classes": classes, "samples": samples,
        "ignoreIndex": 255, "ignoreSourceIds": sorted(ignored),
        "report": {"counts": counts, "totalSamples": len(samples),
                   "skippedWithoutSemanticMask": skipped,
                   "classDistribution": distribution, "downloadProvenance": source_record,
                   "ignoredLabelCounts": ignored_counts, "excludedFrames": excluded_report,
                   "ignorePolicy": "Configured source IDs are excluded from supervision and metrics; their source semantics are not asserted",
                   "splitVideoIds": {split: sorted(splits[split]) for split in split_names},
                   "observedSourceIds": sorted(observed_labels), "fps": float(fps),
                   "fpsSource": "explicit_configuration",
                   "timestampBasis": "frame number divided by configured fps; filename timebase requires verification before playback",
                   "annotationSource": "Endoscapes semantic segmentation annotations"},
    }


def render_preview(manifest: dict[str, Any], output: Path, split: str = "train", limit: int = 6) -> Path:
    """Write a contact sheet of actual images and supplied semantic annotations.

    This is data inspection, not model inference or content review. Colors come
    from the shared anatomy catalog; PNG source IDs come from the manifest.
    """
    if split not in ("train", "val", "test") or type(limit) is not int or limit < 1:
        raise DatasetError("Preview requires split train/val/test and a positive integer limit")
    samples = [sample for sample in manifest["samples"] if sample["split"] == split][:limit]
    if not samples:
        raise DatasetError(f"No samples available for preview split {split!r}")
    catalog_path = Path(__file__).resolve().parents[3] / "contracts/anatomy.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError(f"Cannot load shared anatomy colors from {catalog_path}: {error}") from error
    palette: dict[int, tuple[int, int, int]] = {}
    for item in manifest["classes"]:
        if item["structureId"] == "background":
            continue
        hex_color = catalog[item["structureId"]]["color"].lstrip("#")
        palette[item["sourceId"]] = tuple(int(hex_color[index:index + 2], 16) for index in (0, 2, 4))
    panel_width, panel_height, caption_height = 480, 280, 38
    header_height, footer_height = 54, 44
    sheet = Image.new("RGB", (panel_width * 2, header_height + len(samples) * (panel_height + caption_height) + footer_height), "#101720")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"Endoscapes-Seg50 | {split} | Dataset annotations (not model predictions)", fill="white")
    draw.text((12, 30), "Original image", fill="#cbd5e1")
    draw.text((panel_width + 12, 30), "Semantic annotation overlay", fill="#cbd5e1")
    for row, sample in enumerate(samples):
        rgb = read_image(Path(sample["imagePath"]))
        mask = read_semantic_mask(Path(sample["maskPath"]))
        encode_mask(mask, manifest["classes"], ignore_source_ids=manifest.get("ignoreSourceIds", []))
        overlay = rgb.astype(np.float32)
        for source_id, color in palette.items():
            selected = mask == source_id
            overlay[selected] = 0.45 * overlay[selected] + 0.55 * np.array(color)
        ignored_pixels = np.isin(mask, manifest.get("ignoreSourceIds", []))
        overlay[ignored_pixels] = np.array([150, 150, 150])
        y = header_height + row * (panel_height + caption_height)
        for column, array in enumerate((rgb, overlay.astype(np.uint8))):
            panel = Image.fromarray(array)
            panel.thumbnail((panel_width - 12, panel_height - 6), Image.Resampling.LANCZOS)
            sheet.paste(panel, (column * panel_width + (panel_width - panel.width) // 2, y + (panel_height - panel.height) // 2))
        caption = f"Case {sample['videoId']} | frame {sample['frameNumber']} | {sample['timestampMs']:.0f} ms"
        draw.text((12, y + panel_height + 5), caption, fill="white")
    legend_y = sheet.height - footer_height + 5
    legend_x = 12
    for item in manifest["classes"]:
        structure = item["structureId"]
        if structure == "background":
            continue
        label = catalog[structure]["label"]
        draw.rectangle((legend_x, legend_y, legend_x + 8, legend_y + 8), fill=palette[item["sourceId"]])
        draw.text((legend_x + 12, legend_y - 2), label, fill="white")
        legend_x += int(draw.textlength(label)) + 28
        if legend_x > sheet.width - 220:
            legend_x = 12
            legend_y += 18
    if manifest.get("ignoreSourceIds"):
        draw.rectangle((legend_x, legend_y, legend_x + 8, legend_y + 8), fill=(150, 150, 150))
        draw.text((legend_x + 12, legend_y - 2), "Ignored/unmapped pixels", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")
    return output.resolve()
