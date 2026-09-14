"""Prepare reviewed partial semantic targets without starting model training.

The review PNG convention (255 positive / 0 unknown) is NOT the training
convention. This adapter reads validated review RLE, unions same-class masks,
assigns explicit manifest source IDs only where exactly one class is present,
and writes 255 for conflicting or unreviewed pixels. Original reviews and the
base dataset remain untouched. Pillow and NumPy suffice; no model is loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image


_SPEC = importlib.util.spec_from_file_location("holospex_review_io", Path(__file__).with_name("review_io.py"))
review_io = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(review_io)
ALL_CLASSES = review_io.STRUCTURES | {"background", "gallbladder", "tool"}
SPLITS = ("train", "val", "test")
IGNORE = 255


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _load(path):
    raw = Path(path).read_bytes()
    value, digest = review_io._decode_json(raw)
    review_io._json_values(value, "input JSON")
    return value, raw, digest


def _video_id(value):
    _require((type(value) is int and value >= 0) or
             (isinstance(value, str) and value.isascii() and value.isdigit()),
             "Video IDs must be nonnegative integers or decimal strings")
    return str(int(value))


def _validate_base(base):
    _require(type(base) is dict and base.get("schemaVersion") == "1.0.0", "Unsupported base manifest")
    classes = base.get("classes")
    _require(type(classes) is list and len(classes) == 7, "Base manifest must map all seven classes")
    source_ids, structures = set(), set()
    for index, entry in enumerate(classes):
        _require(type(entry) is dict and type(entry.get("index")) is int and entry["index"] == index,
                 "Base class indices must be contiguous and ordered from zero")
        source, structure = entry.get("sourceId"), entry.get("structureId")
        _require(type(source) is int and 0 <= source < IGNORE and source not in source_ids,
                 "Base source IDs must be unique uint8 labels excluding 255")
        _require(isinstance(structure, str) and structure in ALL_CLASSES and structure not in structures,
                 "Base anatomy IDs must map all seven supported classes exactly once")
        source_ids.add(source)
        structures.add(structure)
    _require(classes[0]["structureId"] == "background" and classes[0]["sourceId"] == 0,
             "Base background must use model index 0 and source ID 0")
    _require(type(base.get("ignoreIndex")) is int and base["ignoreIndex"] == IGNORE,
             "Base ignored target index must be 255")
    ignored = base.get("ignoreSourceIds")
    _require(type(ignored) is list and all(type(v) is int and 0 <= v <= 255 for v in ignored)
             and len(set(ignored)) == len(ignored) and IGNORE in ignored and not (set(ignored) & source_ids),
             "Base ignoreSourceIds must explicitly contain 255 and exclude known class IDs")
    samples = base.get("samples")
    _require(type(samples) is list and bool(samples), "Base samples must be nonempty")
    cases, frames = {}, set()
    for sample in samples:
        _require(type(sample) is dict and sample.get("split") in SPLITS, "Invalid base sample split")
        case = _video_id(sample.get("videoId"))
        _require(case not in cases or cases[case] == sample["split"], "Base video appears in multiple splits")
        cases[case] = sample["split"]
        frame = sample.get("frameNumber")
        _require(type(frame) is int and frame >= 0, "Base frame numbers must be nonnegative integers")
        _require((case, frame) not in frames, "Duplicate base case/frame identity")
        frames.add((case, frame))
        review_io._number(sample.get("timestampMs"), "Base sample timestampMs", 0)
        for field in ("imagePath", "maskPath"):
            value = sample.get(field)
            _require(isinstance(value, str) and Path(value).is_absolute() and Path(value).is_file(),
                     f"Base {field} must identify an existing absolute file")
    return {entry["structureId"]: entry["sourceId"] for entry in classes}, cases


def _validate_resolution(record, bundle, review, bundle_sha, review_sha, base_sha):
    review_io._fields(record, {
        "formatVersion", "artifactType", "bundleId", "bundleSha256", "reviewSha256", "baseManifestSha256",
        "resolvedAt", "resolvedBy", "overlapPolicy", "unknownPixelPolicy", "noteClarifications",
        "authorization", "evidence",
    }, "resolution record")
    _require(record["formatVersion"] == "1.0.0" and record["artifactType"] == "review_training_resolution",
             "Unsupported resolution record")
    _require(record["bundleId"] == bundle["bundleId"] and record["bundleSha256"] == bundle_sha
             and record["reviewSha256"] == review_sha and record["baseManifestSha256"] == base_sha,
             "Resolution record does not match the exact bundle, review and base manifest bytes")
    review_io._utc(record["resolvedAt"], "resolution.resolvedAt")
    _require(record["resolvedBy"] == "project_lead", "Resolution must be recorded by project_lead")
    _require(record["overlapPolicy"] == "ignore_conflicts" and record["unknownPixelPolicy"] == "ignore",
             "Explicit ignore-conflicts and ignore-unknown policies are required")
    _require(record["authorization"] == "prepare_partial_training_targets", "Target preparation authorization is required")
    review_io._fields(record["evidence"], {"source", "text"}, "resolution.evidence")
    _require(record["evidence"]["source"] == "user_message", "Resolution must retain user-message evidence")
    _require(isinstance(record["evidence"]["text"], str) and record["evidence"]["text"].strip(),
             "Resolution must retain the user's nonempty exact clarification text")
    clarifications = record["noteClarifications"]
    _require(type(clarifications) is list, "noteClarifications must be an array")
    decisions = {entry["candidateId"]: entry for entry in review["decisions"]}
    seen = set()
    for clarification in clarifications:
        review_io._fields(clarification, {"candidateId", "interpretation", "reviewedMaskConfirmed"}, "note clarification")
        candidate = clarification["candidateId"]
        _require(isinstance(candidate, str) and candidate in decisions and candidate not in seen,
                 "Unknown or duplicate clarification candidate")
        _require(decisions[candidate]["decision"] == "edited", "A clarified fixed mask must have an edited approval")
        _require(clarification["interpretation"] == "described_original_proposal"
                 and clarification["reviewedMaskConfirmed"] is True,
                 "Clarification must confirm the reviewed mask and describe the original-proposal note")
        seen.add(candidate)


def _read_base_mask(sample):
    image_raw = Path(sample["imagePath"]).read_bytes()
    mask_raw = Path(sample["maskPath"]).read_bytes()
    with Image.open(io.BytesIO(image_raw)) as image:
        dimensions = image.size
        image.verify()
    with Image.open(io.BytesIO(mask_raw)) as source:
        _require(source.format == "PNG", "Base semantic masks must be lossless PNG files")
        mask = np.array(source)
    if mask.ndim == 3 and mask.shape[2] == 3:
        _require(np.array_equal(mask[:, :, 0], mask[:, :, 1]) and np.array_equal(mask[:, :, 0], mask[:, :, 2]),
                 "Base RGB mask channels must contain identical class IDs")
        mask = mask[:, :, 0]
    _require(mask.ndim == 2 and np.issubdtype(mask.dtype, np.integer)
             and (mask.shape[1], mask.shape[0]) == dimensions, "Base image/mask shape or dtype mismatch")
    return mask, {"imagePath": sample["imagePath"], "imageSha256": _sha(image_raw),
                  "maskPath": sample["maskPath"], "maskSha256": _sha(mask_raw)}


def _new_statistics(classes, ignored):
    return {
        "counts": {split: 0 for split in SPLITS},
        "classDistribution": {split: {entry["structureId"]: {"images": 0, "pixels": 0} for entry in classes} for split in SPLITS},
        "ignoredLabelCounts": {split: {"images": 0, "pixels": 0,
            "bySourceId": {str(value): {"images": 0, "pixels": 0} for value in ignored}} for split in SPLITS},
        "observedSourceIds": set(), "splitVideoIds": {split: set() for split in SPLITS},
    }


def _count_mask(stats, mask, sample, classes, ignored):
    mapping = {entry["sourceId"]: entry["structureId"] for entry in classes}
    values, counts = np.unique(mask, return_counts=True)
    _require(set(map(int, values)) <= set(mapping) | set(ignored), "Mask contains an unmapped source label")
    split = sample["split"]
    stats["counts"][split] += 1
    stats["splitVideoIds"][split].add(int(_video_id(sample["videoId"])))
    ignored_pixels = 0
    for value, count in zip(values, counts):
        value, count = int(value), int(count)
        stats["observedSourceIds"].add(value)
        if value in ignored:
            ignored_pixels += count
            record = stats["ignoredLabelCounts"][split]["bySourceId"][str(value)]
        else:
            record = stats["classDistribution"][split][mapping[value]]
        record["images"] += 1
        record["pixels"] += count
    if ignored_pixels:
        stats["ignoredLabelCounts"][split]["images"] += 1
        stats["ignoredLabelCounts"][split]["pixels"] += ignored_pixels


def prepare_training(bundle_path, review_path, base_manifest_path, resolution_record_path, output_dir):
    """Publish a new combined manifest and immutable reviewed partial targets.

    Returns ``{manifestPath, preparationPath, summaryPath, summary}``. Entire inputs, source
    files, class/split mapping and all derived native targets validate before
    any output is created. Images with all-ignored targets are listed but not
    added. An expansion with no positive supervision fails without publishing.
    Base sample dictionaries and their relative order are preserved exactly.
    """
    destination = Path(output_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite prepared targets: {destination}")
    bundle, bundle_raw, bundle_sha = _load(bundle_path)
    review, review_raw, review_sha = _load(review_path)
    base, base_raw, base_sha = _load(base_manifest_path)
    resolution, resolution_raw, resolution_sha = _load(resolution_record_path)
    validation = review_io.validate_review(review, bundle, bundle_sha)
    _require(review["reviewer"]["reviewScope"] == "anatomy", "Only anatomy-scope reviews can prepare training targets")
    source_ids, base_cases = _validate_base(base)
    _validate_resolution(resolution, bundle, review, bundle_sha, review_sha, base_sha)
    eligible = set(validation["eligibleCandidateIds"])
    _require(bool(eligible), "Review has no eligible accepted/edited anatomy candidates")
    for image in bundle["images"]:
        _require(image["split"] == "train", "Only TRAIN review images may be added")
        _require(_video_id(image["videoId"]) not in base_cases, "Review case overlaps an existing base train/val/test case")
    review_io.validate_bundle_files(bundle, bundle_path)

    # Fixed names only: a dataset license is copied as original bytes, never
    # obtained from a filename supplied by review notes or other free text.
    source_dir = Path(bundle_path).resolve().parent
    licenses = []
    for name in ("Endoscapes_LICENSE.txt", "SAM2_LICENSE.txt"):
        path = source_dir / "licenses" / name
        if path.is_file():
            _require(path.resolve().is_relative_to(source_dir), "Source license escapes bundle directory")
            raw = path.read_bytes()
            licenses.append((name, raw))

    classes, ignored = base["classes"], base["ignoreSourceIds"]
    stats = _new_statistics(classes, ignored)
    base_files = []
    for sample in base["samples"]:
        mask, audit = _read_base_mask(sample)
        _count_mask(stats, mask, sample, classes, ignored)
        base_files.append(audit)
    decisions = {entry["candidateId"]: entry for entry in review["decisions"]}
    prepared, image_records, new_samples = [], [], []
    totals = Counter(reviewedForegroundUnionPixels=0, conflictingPixels=0, retainedPixels=0, unreviewedPixels=0)
    retained_by_class = {structure: 0 for structure in sorted(review_io.STRUCTURES)}
    report = base.get("report", {})
    fps = report.get("fps") if isinstance(report, dict) else None
    if fps is not None:
        review_io._number(fps, "base.report.fps", 0)
        _require(fps > 0, "Configured base fps must be positive")
    for image in bundle["images"]:
        shape = (image["height"], image["width"])
        by_class = {structure: np.zeros(shape, dtype=bool) for structure in sorted(review_io.STRUCTURES)}
        candidate_records = []
        for candidate in image["candidates"]:
            decision = decisions.get(candidate["id"])
            is_eligible = candidate["id"] in eligible
            candidate_record = {"candidateId": candidate["id"], "structureId": candidate["structureId"],
                "sourceAnnotationId": candidate["sourceAnnotationId"], "proposalMaskSha256": candidate["maskSha256"],
                "decision": decision["decision"] if decision else "missing", "eligible": is_eligible}
            if decision:
                pixels = review_io.decode_rle(decision["mask"])
                candidate_record.update(reviewedMaskSha256=_sha(pixels), notes=decision["notes"],
                    reviewedAt=decision["reviewedAt"], reviewMilliseconds=decision["reviewMilliseconds"])
                if is_eligible:
                    by_class[candidate["structureId"]] |= np.frombuffer(pixels, dtype=np.uint8).reshape(shape).astype(bool)
            candidate_records.append(candidate_record)
        coverage = np.stack(list(by_class.values())).sum(axis=0)
        union, conflict, unique = coverage > 0, coverage > 1, coverage == 1
        target = np.full(shape, IGNORE, dtype=np.uint8)
        per_class = {}
        for structure, pixels in by_class.items():
            scored = pixels & unique
            target[scored] = source_ids[structure]
            per_class[structure] = {"sourceId": source_ids[structure], "reviewedForegroundPixels": int(pixels.sum()),
                "ignoredConflictPixels": int((pixels & conflict).sum()), "retainedPixels": int(scored.sum())}
        values, counts = np.unique(target, return_counts=True)
        _require(set(map(int, values)) <= {source_ids[s] for s in review_io.STRUCTURES} | {IGNORE}
                 and not np.any(target == 0), "Derived targets must contain only focus source IDs and ignored 255, never background")
        retained = int(unique.sum())
        record = {"imageId": image["id"], "videoId": image["videoId"], "frameNumber": image["frameNumber"],
            "split": "train", "width": image["width"], "height": image["height"], "imageSha256": image["imageSha256"],
            "reviewer": review["reviewer"], "bundleSha256": bundle_sha, "reviewSha256": review_sha,
            "resolutionRecordSha256": resolution_sha, "candidates": candidate_records,
            "included": retained > 0, "reviewedForegroundUnionPixels": int(union.sum()),
            "conflictingPixels": int(conflict.sum()), "retainedPixels": retained,
            "unreviewedPixels": int((coverage == 0).sum()), "ignoredPixels": int((target == IGNORE).sum()),
            "classPixelCounts": per_class, "nativeTargetPixelsSha256": _sha(target.tobytes(order="C")),
            "sourceIdPixelCounts": {str(int(v)): int(c) for v, c in zip(values, counts)},
            "overlapPolicy": "ignore_conflicts", "unknownPixelPolicy": "ignore",
            "sameClassPolicy": "union", "targetEncoding": "uint8 semantic source IDs; 255 ignored; no background labels"}
        if not retained:
            record["omissionReason"] = "no_eligible_candidates" if not any(c["eligible"] for c in candidate_records) else "all_eligible_pixels_conflict"
        else:
            source_path = (source_dir / image["imagePath"]).resolve()
            _require(source_path.is_relative_to(source_dir), "Source image escapes bundle directory")
            source_raw = source_path.read_bytes()
            _require(_sha(source_raw) == image["imageSha256"], "Source image changed after validation")
            suffix = source_path.suffix.lower()
            _require(suffix in {".jpg", ".jpeg", ".png"}, "Reviewed source images must be JPEG or PNG")
            image_relative, mask_relative = f"images/{image['id']}{suffix}", f"masks/{image['id']}.png"
            record.update(imagePath=image_relative, maskPath=mask_relative)
            sample = {"split": "train", "imagePath": str(destination / image_relative),
                "maskPath": str(destination / mask_relative), "videoId": int(_video_id(image["videoId"])),
                "frameNumber": image["frameNumber"], "timestampMs": image["frameNumber"] / fps * 1000 if fps else 0.0,
                "annotationSource": "reviewed_partial_anatomy", "reviewProvenancePath": str(destination / f"provenance/{image['id']}.json")}
            record["timestampBasis"] = "source frame number divided by explicit base fps; not verified playback timing" if fps else "independent still image; zero timestamp does not establish source video timing"
            new_samples.append(sample)
            prepared.append((record, target, source_raw))
            _count_mask(stats, target, sample, classes, ignored)
        image_records.append(record)
        totals.update({key: record[key] for key in totals})
        for structure, counts in per_class.items():
            retained_by_class[structure] += counts["retainedPixels"]
    _require(bool(prepared), "No retained positive supervision remains after conflict handling")

    combined = copy.deepcopy(base)
    combined["dataset"] = f"{base.get('dataset', 'endoscapes')}+reviewed-partial"
    combined["root"] = str(destination)
    combined["samples"] = copy.deepcopy(base["samples"]) + new_samples
    stats["observedSourceIds"] = sorted(stats["observedSourceIds"])
    stats["splitVideoIds"] = {split: sorted(videos) for split, videos in stats["splitVideoIds"].items()}
    stats["totalSamples"] = len(combined["samples"])
    combined["report"] = {**stats, "baseReport": copy.deepcopy(base.get("report", {})),
        "annotationSource": "Original Endoscapes semantic masks plus explicitly reviewed partial TRAIN masks",
        "ignorePolicy": "Explicit configured source IDs ignored; new partial targets mark uncovered and conflicting pixels 255, never background",
        "partialImageCount": len(new_samples), "fps": fps,
        "timestampBasis": "Original base samples retained exactly; added stills reuse explicit base fps only when available, never verified video synchronization"}
    bindings = {"bundleSha256": bundle_sha, "reviewSha256": review_sha, "baseManifestSha256": base_sha,
                "resolutionRecordSha256": resolution_sha}
    combined["reviewedPartialProvenance"] = {**bindings, "bundleId": bundle["bundleId"], "reviewer": review["reviewer"],
        "preparationPath": str(destination / "summary.json"), "overlapPolicy": "ignore_conflicts",
        "unknownPixelPolicy": "ignore", "trainingStarted": False}
    for split in ("val", "test"):
        _require([s for s in combined["samples"] if s["split"] == split]
                 == [s for s in base["samples"] if s["split"] == split], "Held-out samples changed unexpectedly")
    summary = {"baseSampleCount": len(base["samples"]), "reviewCandidateCount": validation["candidateCount"],
        "eligibleCandidateCount": len(eligible), "addedTrainImageCount": len(new_samples),
        "omittedImageCount": len(bundle["images"]) - len(new_samples), "combinedCounts": stats["counts"],
        **dict(totals), "retainedPixelsByClass": retained_by_class, "trainingStarted": False}
    preparation = {"formatVersion": "1.0.0", "artifactType": "reviewed_partial_training_preparation",
        "createdAt": datetime.now(timezone.utc).isoformat(), **bindings, "bundleId": bundle["bundleId"],
        "preparerCodeSha256": _sha(Path(__file__).read_bytes()),
        "reviewer": review["reviewer"], "resolution": resolution, "summary": summary, "images": image_records,
        "baseFiles": base_files, "classes": copy.deepcopy(classes), "ignoreSourceIds": copy.deepcopy(ignored),
        "trainingStarted": False, "humanDecisionsChanged": False,
        "sourcePaths": {"bundle": str(Path(bundle_path).resolve()), "review": str(Path(review_path).resolve()),
            "baseManifest": str(Path(base_manifest_path).resolve()), "resolutionRecord": str(Path(resolution_record_path).resolve())},
        "sourceBundleSnapshotMeaning": "provenance/source-bundle.json preserves exact original bundle bytes; its image paths remain relative to the original source bundle directory, not this provenance directory",
        "licenses": [{"path": f"licenses/{name}", "sha256": _sha(raw)} for name, raw in licenses],
        "manifestPath": "manifest.json", "policy": {"sameClass": "union", "differentClassOverlap": "ignore255",
            "unreviewedPixels": "ignore255", "backgroundPixelsAdded": 0, "allIgnoreImages": "omit_and_record"}}

    # Validate/assemble every target before any output path is created. Publish
    # only a complete staged directory into an exclusively reserved destination.
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    reserved = False
    try:
        for directory in ("images", "masks", "provenance"):
            (stage / directory).mkdir()
        if licenses:
            (stage / "licenses").mkdir()
            for name, raw in licenses:
                (stage / "licenses" / name).write_bytes(raw)
        inputs = {"source-bundle": bundle_raw, "source-review": review_raw, "base-manifest": base_raw, "resolution": resolution_raw}
        for name, raw in inputs.items():
            (stage / f"provenance/{name}.json").write_bytes(raw)
        for record, target, source_raw in prepared:
            (stage / record["imagePath"]).write_bytes(source_raw)
            mask_path = stage / record["maskPath"]
            Image.fromarray(target).save(mask_path)
            record["targetPngSha256"] = _sha(mask_path.read_bytes())
            with Image.open(mask_path) as saved:
                _require(saved.mode == "L" and np.array_equal(np.asarray(saved), target), "Saved target PNG differs from assembled source labels")
        for record in image_records:
            (stage / f"provenance/{record['imageId']}.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        manifest_raw = (json.dumps(combined, indent=2, allow_nan=False) + "\n").encode("utf-8")
        (stage / "manifest.json").write_bytes(manifest_raw)
        preparation["combinedManifestSha256"] = _sha(manifest_raw)
        readme = "\n".join([
            "# Reviewed partial Endoscapes training targets", "",
            "These derived masks preserve the named reviewer's accepted/edited decisions and the project lead's recorded resolution. No training is started by this preparation.", "",
            "`manifest.json` combines the unchanged base samples with new TRAIN samples. Base files remain at their original absolute paths. Validation and test sample dictionaries/order are unchanged.", "",
            "`masks/` contains uint8 semantic source IDs from the manifest's explicit class map. Source 255 means ignored: conflicting and unreviewed pixels are excluded. No background pixels are fabricated. Same-class candidate masks are unioned; different-class overlaps are ignored. These PNGs differ from the imported review PNG convention, where 255 meant foreground.", "",
            "`summary.json` records source hashes, native pixel counts, omitted images, image/target provenance and an artifact inventory. `provenance/` preserves exact input JSON bytes and per-image records. The source-bundle snapshot still references images relative to its original source directory recorded in the summary; it is not a standalone copied proposal bundle.", "",
            "Source: [Endoscapes / CAMMA](https://github.com/CAMMA-public/Endoscapes). The source release states CC BY-NC-SA 4.0 for non-commercial scientific research. Retain attribution and these terms with derived images/masks. Source SAM 2 proposals use the official [SAM 2 repository](https://github.com/facebookresearch/sam2), whose notice is Apache-2.0.", "",
            "Original Endoscapes and SAM license files are copied to `licenses/` when present beside the source bundle; their exact hashes appear in the summary. This output does not change the terms of the source data or establish anatomical correctness.", "",
        ])
        (stage / "README.md").write_text(readme)
        preparation["artifactInventory"] = [{"path": str(path.relative_to(stage)), "sha256": _sha(path.read_bytes())}
            for path in sorted(stage.rglob("*")) if path.is_file()]
        preparation["artifactInventoryScope"] = "All prepared files except summary.json itself"
        (stage / "summary.json").write_text(json.dumps(preparation, indent=2, allow_nan=False) + "\n")
        destination.mkdir()
        reserved = True
        stage.rename(destination)
        reserved = False
    except BaseException:
        if reserved:
            destination.rmdir()
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"manifestPath": str(destination / "manifest.json"),
            "preparationPath": str(destination / "summary.json"), "summaryPath": str(destination / "summary.json"), "summary": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("bundle", "review", "base-manifest", "resolution-record", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_training(args.bundle, args.review, args.base_manifest, args.resolution_record, args.output_dir)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
