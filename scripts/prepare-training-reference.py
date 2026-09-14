"""Prepare a small, train-only browser reference pack from an existing ML manifest.

Requires Pillow only; no model imports, downloads, inference or test-split fallback.
Original JPEG/source-mask bytes are preserved. The index PNG remaps documented
source IDs without resizing and retains ignored pixels as 255.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 128 * 1024 * 1024
STRUCTURES = ("background", "gallbladder", "cystic_duct", "cystic_artery", "cystic_plate",
              "hepatocystic_triangle_dissection", "tool")
SOURCE_IDS = (0, 5, 4, 3, 1, 2, 6)
CLASSES = [{"index": index, "sourceId": source, "structureId": structure}
           for index, (source, structure) in enumerate(zip(SOURCE_IDS, STRUCTURES))]
SOURCE_URL = "https://github.com/CAMMA-public/Endoscapes"


class ReferenceError(ValueError):
    pass


def integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ReferenceError(f"{name} must be a nonnegative integer.")
    return value


def logical_path(value: Any):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ReferenceError("Manifest source paths must be nonempty strings.")
    path = PureWindowsPath(value) if "\\" in value or re.match(r"^[A-Za-z]:", value) else PurePosixPath(value)
    if ".." in path.parts:
        raise ReferenceError("Manifest source paths cannot contain parent traversal.")
    return path


def resolve_source(value: Any, original_root: Any, local_root: Path) -> Path:
    path, recorded_root = logical_path(value), logical_path(original_root)
    if not recorded_root.is_absolute():
        raise ReferenceError("Manifest root must be absolute; regenerate the ML manifest.")
    try:
        relative = path.relative_to(recorded_root) if path.is_absolute() else path
    except ValueError as cause:
        raise ReferenceError("Source path is outside the manifest root; relocation would be ambiguous.") from cause
    resolved = local_root.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(local_root) or not resolved.is_file():
        raise ReferenceError(f"Missing or out-of-root source file: {relative}. Supply --data-root for a relocated dataset.")
    return resolved


def validate_manifest(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("schemaVersion") != "1.0.0" or value.get("dataset") != "endoscapes-seg50":
        raise ReferenceError("Expected the existing Endoscapes-Seg50 training manifest.")
    classes = value.get("classes")
    if not isinstance(classes, list) or len(classes) != 7 or any(not isinstance(item, dict) for item in classes):
        raise ReferenceError("Manifest classes must match the shared Endoscapes source/index mapping.")
    for item in classes:
        integer(item.get("index"), "Class index"); integer(item.get("sourceId"), "Class source ID")
    if sorted(classes, key=lambda item: item["index"]) != CLASSES:
        raise ReferenceError("Manifest classes must match the shared Endoscapes source/index mapping.")
    if value.get("ignoreIndex") != 255 or value.get("ignoreSourceIds") not in ([], [255]):
        raise ReferenceError("Only explicitly configured source ID 255 may be ignored.")
    report = value.get("report")
    splits = report.get("splitVideoIds") if isinstance(report, dict) else None
    if not isinstance(splits, dict) or any(not isinstance(splits.get(split), list) for split in ("train", "val", "test")):
        raise ReferenceError("Manifest must provide explicit train/val/test video split membership.")
    membership = {}
    for split in ("train", "val", "test"):
        videos = splits[split]
        for video in videos:
            integer(video, "Split video ID")
            if video in membership:
                raise ReferenceError("Duplicate video or leakage across manifest splits.")
            membership[video] = split
    entries = value.get("samples")
    if not isinstance(entries, list):
        raise ReferenceError("Manifest samples must be an array.")
    seen, training = set(), []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ReferenceError("Manifest sample must be an object.")
        video = integer(entry.get("videoId"), "Sample video ID")
        number = integer(entry.get("frameNumber"), "Sample frame number")
        if entry.get("split") not in ("train", "val", "test") or membership.get(video) != entry["split"]:
            raise ReferenceError("Sample split contradicts explicit manifest video membership.")
        identity = (video, number)
        if identity in seen:
            raise ReferenceError("Duplicate video/frame identity in manifest.")
        seen.add(identity)
        if entry["split"] == "train":
            training.append(entry)
    if not training:
        raise ReferenceError("No explicitly identified training samples are available.")
    return sorted(training, key=lambda entry: (entry["videoId"], entry["frameNumber"]))


def prepare_case(entry: dict[str, Any], manifest: dict[str, Any], local_root: Path) -> dict[str, bytes]:
    identity = f"{entry['videoId']}_{entry['frameNumber']}"
    image_path = resolve_source(entry.get("imagePath"), manifest["root"], local_root)
    mask_path = resolve_source(entry.get("maskPath"), manifest["root"], local_root)
    if image_path.stem != identity or mask_path.stem != identity:
        raise ReferenceError(f"Source filenames contradict video/frame identity {identity}.")
    if image_path.stat().st_size + mask_path.stat().st_size > MAX_BYTES:
        raise ReferenceError("Source pair exceeds the 128 MiB reference-pack limit.")
    image_bytes, source_bytes = image_path.read_bytes(), mask_path.read_bytes()
    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
        if image.format != "JPEG" or not (0 < width <= 8192 and 0 < height <= 8192) or width * height > 16_000_000:
            raise ReferenceError(f"{identity}: expected an original JPEG within supported image dimensions.")
        if image.getexif().get(274, 1) != 1:
            raise ReferenceError(f"{identity}: EXIF orientation requires an explicit image/label conversion first.")
        image.load()
    with Image.open(BytesIO(source_bytes)) as mask:
        if mask.format != "PNG" or mask.size != (width, height):
            raise ReferenceError(f"{identity}: source mask PNG dimensions do not match its image.")
        if mask.mode in ("L", "P"):
            values = mask.tobytes()  # Palette indices are IDs; never convert palette colors to grayscale.
        elif mask.mode == "RGB":
            rgb = mask.tobytes()
            if rgb[0::3] != rgb[1::3] or rgb[0::3] != rgb[2::3]:
                raise ReferenceError(f"{identity}: RGB mask channels differ; expected repeated label IDs.")
            values = rgb[0::3]
        else:
            raise ReferenceError(f"{identity}: unsupported mask mode {mask.mode}; expected 8-bit grayscale, palette or repeated RGB IDs.")
    counts = Counter(values)
    ignored = set(manifest["ignoreSourceIds"])
    unknown = set(counts) - set(SOURCE_IDS) - ignored
    if unknown:
        raise ReferenceError(f"{identity}: unknown source mask IDs {sorted(unknown)}.")
    lookup = bytearray(256)
    for item in CLASSES:
        lookup[item["sourceId"]] = item["index"]
    lookup[255] = 255
    encoded = values.translate(bytes(lookup))
    buffer = BytesIO()
    Image.frombytes("L", (width, height), encoded).save(buffer, format="PNG")
    index_bytes = buffer.getvalue()
    labels = {
        "artifactType": "dataset_annotation_sample",
        "annotationSource": {"dataset": "Endoscapes-Seg50", "kind": "supplied_dataset_annotation", "split": "train",
                             "sourceUrl": SOURCE_URL, "videoId": entry["videoId"], "sourceFrameNumber": entry["frameNumber"]},
        "frame": {"mediaId": f"endoscapes-still-{identity}", "frameNumber": 0, "timestampMs": 0,
                  "width": width, "height": height, "coordinateSpace": "original_pixels"},
        "classes": CLASSES,
        "raster": {"dtype": "uint8", "shape": [height, width], "ignoreValue": 255,
                   "pixelCounts": {item["structureId"]: counts[item["sourceId"]] for item in CLASSES[1:]},
                   "ignoredPixelCount": counts[255]},
        "structures": [],
        "conversion": {"geometry": "No approximate polygons generated. Use the exact full-resolution index mask, including holes and disconnected regions.",
                       "visibility": "Supplied training annotations; no visibility inference or answer review performed.", "withheldComponents": {}},
        "filesSha256": {"image.jpg": sha256(image_bytes).hexdigest(), "labels-source.png": sha256(source_bytes).hexdigest(),
                        "labels-index.png": sha256(index_bytes).hexdigest()},
    }
    return {f"{identity}/image.jpg": image_bytes, f"{identity}/labels-source.png": source_bytes,
            f"{identity}/labels-index.png": index_bytes, f"{identity}/labels.json": json_bytes(labels)}


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def prepare_training_reference(manifest_path: Path, output: Path, limit: int = 12, data_root: Path | None = None,
                               offset: int = 0) -> list[str]:
    if type(limit) is not int or not 1 <= limit <= 40:
        raise ReferenceError("Choose --limit between 1 and 40 training frames.")
    integer(offset, "Batch offset")
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ReferenceError(f"Training manifest is missing: {manifest_path}. Supply --manifest for Person 1's prepared training data.")
    if manifest_path.stat().st_size > 16 * 1024 * 1024:
        raise ReferenceError("Training manifest exceeds 16 MiB.")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    training = validate_manifest(manifest)
    if offset >= len(training):
        raise ReferenceError("Batch offset is beyond the available training frames.")
    recorded_root = logical_path(manifest.get("root"))
    if not recorded_root.is_absolute():
        raise ReferenceError("Manifest root must be an absolute dataset directory.")
    local_root = (data_root.expanduser() if data_root is not None else Path(str(recorded_root))).resolve()
    if not local_root.is_dir():
        raise ReferenceError("Training data root is unavailable here. Supply --data-root for the local Endoscapes directory.")
    if output.expanduser().is_symlink():
        raise ReferenceError("Output cannot be a linked directory.")
    destination = output.expanduser().resolve()
    if destination == local_root or local_root.is_relative_to(destination) or destination == ROOT:
        raise ReferenceError("Choose a separate generated output directory, not the dataset or repository root.")
    files: dict[str, bytes] = {}
    selected = training[offset:offset + limit]
    for entry in selected:
        files.update(prepare_case(entry, manifest, local_root))
        if sum(map(len, files.values())) > MAX_BYTES:
            raise ReferenceError("Reference pack exceeds 128 MiB. Choose a smaller --limit.")
    identities = [f"{entry['videoId']}_{entry['frameNumber']}" for entry in selected]
    files["sample-index.json"] = json_bytes({
        "artifactType": "training_reference_pack", "schemaVersion": 1, "dataset": "Endoscapes-Seg50", "split": "train",
        "sourceUrl": SOURCE_URL, "license": "CC BY-NC-SA 4.0", "credit": "CAMMA / Endoscapes authors",
        "manifestSha256": sha256(manifest_bytes).hexdigest(), "availableTrainingFrames": len(training),
        "offset": offset, "limit": limit,
        "selection": "Training batch sorted by video ID and source frame number; not selected by prediction quality.",
        "samples": [{"id": identity, "labels": f"{identity}/labels.json"} for identity in identities],
        "notice": "Supplied training annotations, not reviewed lesson answers or live predictions. Keep this pack local and outside public deployment.",
    })
    if len(files) > 200 or sum(map(len, files.values())) > MAX_BYTES:
        raise ReferenceError("Reference pack exceeds the browser's 200-file / 128 MiB limit.")
    # Preflight every existing path and byte before writing. Never delete source files
    # or silently merge a previous reference selection into a new one.
    if destination.exists():
        if not destination.is_dir():
            raise ReferenceError("Output must be a directory.")
        for path in destination.rglob("*"):
            if path.is_symlink() or not path.resolve().is_relative_to(destination):
                raise ReferenceError("Output contains a linked path; choose a fresh output directory.")
            if path.is_file():
                name = path.relative_to(destination).as_posix()
                if name not in files or path.read_bytes() != files[name]:
                    raise ReferenceError("Output already contains a different reference pack. Choose a fresh --output directory.")
    for name, content in files.items():
        path = destination / name
        if path.exists() and not path.is_file():
            raise ReferenceError("Output contains a directory where a prepared file belongs.")
    for name, content in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return identities


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "ml/outputs/endoscapes-manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/training-reference")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--offset", type=int, default=0, help="First frame in the sorted training batch. Use a fresh --output for a different batch.")
    parser.add_argument("--data-root", type=Path, help="Local replacement for the manifest's absolute dataset root; relative source layout must match.")
    args = parser.parse_args()
    try:
        identities = prepare_training_reference(args.manifest, args.output, args.limit, args.data_root, args.offset)
    except (ReferenceError, OSError, ValueError) as cause:
        parser.exit(1, f"Training reference unavailable: {cause}\n")
    print(f"Prepared {len(identities)} training reference frames in {args.output.resolve()}")
    print("Source images and masks remain local. No model training or inference was performed.")


if __name__ == "__main__":
    main()
