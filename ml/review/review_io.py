"""Validate and import the offline candidate-mask review format.

Validation uses the standard library. Image verification and PNG import need
Pillow (``ml[data]``). Imported masks are separate partial positive annotations:
255 means this candidate's reviewed foreground and 0 means UNKNOWN, never a
dense background target. This module does not enroll data in training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile


VERSION = "1.0.0"
STRUCTURES = {
    "cystic_duct", "cystic_artery", "cystic_plate",
    "hepatocystic_triangle_dissection",
}
DECISIONS = {"pending", "accepted", "edited", "rejected", "needs_expert"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
MAX_PIXELS = 50_000_000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(value, required, location):
    _require(type(value) is dict, f"{location} must be an object")
    _require(set(value) == set(required),
             f"{location} fields differ: missing {sorted(set(required) - set(value))}, "
             f"unexpected {sorted(set(value) - set(required))}")


def _text(value, location, *, safe=False):
    _require(isinstance(value, str) and bool(value.strip()), f"{location} must be nonempty text")
    _require(not any(ord(character) < 32 for character in value), f"{location} contains control characters")
    if safe:
        _require(bool(SAFE_ID.fullmatch(value)) and value not in {".", ".."},
                 f"{location} must be a safe identifier")


def _integer(value, location, minimum=0):
    _require(type(value) is int and value >= minimum, f"{location} must be an integer >= {minimum}")


def _number(value, location, minimum=None):
    _require(type(value) in (int, float), f"{location} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite and (minimum is None or value >= minimum),
             f"{location} must be finite" + (f" and >= {minimum}" if minimum is not None else ""))


def _sha(value, location):
    _require(isinstance(value, str) and bool(SHA256.fullmatch(value)),
             f"{location} must be a lowercase SHA-256 digest")


def _utc(value, location):
    _text(value, location)
    _require(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)", value)),
             f"{location} must be a UTC ISO timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{location} must be a valid UTC ISO timestamp") from error


def _json_values(value, location):
    """Provenance objects are extensible, but must still be finite JSON."""
    if type(value) is dict:
        for key, item in value.items():
            _require(isinstance(key, str), f"{location} keys must be strings")
            _json_values(item, f"{location}.{key}")
    elif type(value) is list:
        for item in value:
            _json_values(item, location)
    elif type(value) in (int, float):
        _number(value, location)
    else:
        _require(value is None or type(value) in (str, bool), f"{location} must contain only JSON values")


def _relative_path(value, location):
    _text(value, location)
    _require("\\" not in value and ":" not in value and not value.startswith("/"),
             f"{location} must be a relative POSIX path")
    _require(all(part not in {"", ".", ".."} for part in value.split("/")),
             f"{location} must not contain empty, dot or parent segments")
    _require(not PurePosixPath(value).is_absolute(), f"{location} must be relative")


def decode_rle(mask):
    """Return canonical row-major uint8 0/1 bytes; reject noncanonical runs."""
    _fields(mask, {"encoding", "width", "height", "counts"}, "mask")
    _require(mask["encoding"] == "rle-row-major-v1", "Unsupported mask encoding")
    _integer(mask["width"], "mask.width", 1)
    _integer(mask["height"], "mask.height", 1)
    total = mask["width"] * mask["height"]
    _require(total <= MAX_PIXELS, "Mask exceeds the review pixel limit")
    counts = mask["counts"]
    _require(type(counts) is list and bool(counts), "mask.counts must be a nonempty array")
    _require(len(counts) <= total + 1, "Too many RLE runs")
    for index, count in enumerate(counts):
        _integer(count, f"mask.counts[{index}]", 0 if index == 0 else 1)
    _require(sum(counts) == total, "RLE run sum must equal width * height")
    # Validate every count before allocating output, including huge integers.
    output = bytearray(total)
    offset = 0
    for index, count in enumerate(counts):
        if index % 2:
            output[offset:offset + count] = b"\x01" * count
        offset += count
    return bytes(output)


def encode_rle(pixels, width=None, height=None):
    """Encode flat 0/1 bytes with dimensions, or a rectangular 2-D array/list.

    NumPy arrays are accepted through ``tolist`` without importing NumPy. Bool
    mask elements are accepted; integer 255 is not silently interpreted as 1.
    """
    if hasattr(pixels, "tolist"):
        pixels = pixels.tolist()
    if width is None and height is None:
        _require(isinstance(pixels, (list, tuple)) and bool(pixels), "Expected a nonempty 2-D mask")
        height = len(pixels)
        _require(isinstance(pixels[0], (list, tuple)), "Flat pixels require width and height")
        width = len(pixels[0])
        _require(all(isinstance(row, (list, tuple)) and len(row) == width for row in pixels),
                 "Mask rows must have equal width")
        values = [pixel for row in pixels for pixel in row]
    else:
        values = list(pixels)
    _integer(width, "width", 1)
    _integer(height, "height", 1)
    _require(width * height <= MAX_PIXELS and len(values) == width * height,
             "Pixel count must equal bounded width * height")
    _require(all(type(value) in (int, bool) and value in (0, 1) for value in values),
             "Mask pixels must be binary 0/1")
    counts, previous, count = [], 0, 0
    for value in values:
        if value == previous:
            count += 1
        else:
            counts.append(count)
            previous, count = value, 1
    counts.append(count)
    return {"encoding": "rle-row-major-v1", "width": width, "height": height, "counts": counts}


def mask_sha256(mask):
    """Digest decoded 0/1 bytes, not JSON or PNG bytes."""
    return hashlib.sha256(decode_rle(mask)).hexdigest()


def validate_bundle(bundle):
    """Validate bundle structure, train-only identities and proposal pixels.

    Returns None. Disk image identities are checked by ``validate_bundle_files``
    and always checked by the CLI and importer. Selection/generator provenance
    are extensible objects, not assertions that this code verifies their truth.
    """
    _fields(bundle, {"formatVersion", "artifactType", "bundleId", "createdAt",
                     "sourceManifestSha256", "selection", "generator", "images", "classes"}, "bundle")
    _require(bundle["formatVersion"] == VERSION and bundle["artifactType"] == "candidate_mask_bundle",
             "Unsupported bundle type/version")
    _text(bundle["bundleId"], "bundleId", safe=True)
    _utc(bundle["createdAt"], "createdAt")
    _sha(bundle["sourceManifestSha256"], "sourceManifestSha256")
    for key in ("selection", "generator"):
        _require(type(bundle[key]) is dict and bool(bundle[key]), f"{key} must be a nonempty provenance object")
        _json_values(bundle[key], key)
    _require(type(bundle["classes"]) is list and bool(bundle["classes"]), "classes must be nonempty")
    class_ids = set()
    for entry in bundle["classes"]:
        _fields(entry, {"structureId", "label", "color"}, "class")
        structure = entry["structureId"]
        _require(isinstance(structure, str) and structure in STRUCTURES and structure not in class_ids,
                 "Classes must have unique supported small-anatomy IDs")
        class_ids.add(structure)
        _text(entry["label"], "class.label")
        _require(isinstance(entry["color"], str) and bool(re.fullmatch(r"#[0-9a-fA-F]{6}", entry["color"])),
                 "Class color must be #RRGGBB")
    _require(type(bundle["images"]) is list and bool(bundle["images"]), "images must be nonempty")
    image_ids, candidate_ids, annotation_ids, image_paths = set(), set(), set(), set()
    for image in bundle["images"]:
        _fields(image, {"id", "videoId", "frameNumber", "split", "width", "height",
                        "imagePath", "imageSha256", "candidates"}, "image")
        _text(image["id"], "image.id", safe=True)
        _require(image["id"] not in image_ids, "Duplicate image ID")
        image_ids.add(image["id"])
        _text(image["videoId"], "image.videoId", safe=True)
        _integer(image["frameNumber"], "image.frameNumber")
        _require(image["id"] == f"{image['videoId']}_{image['frameNumber']}",
                 "Image ID must match source video/frame identity")
        _require(image["split"] == "train", "Only official train images may enter this review bundle")
        _integer(image["width"], "image.width", 1)
        _integer(image["height"], "image.height", 1)
        _require(image["width"] * image["height"] <= MAX_PIXELS, "Image exceeds review pixel limit")
        _relative_path(image["imagePath"], "image.imagePath")
        _require(image["imagePath"] not in image_paths, "Duplicate image path")
        image_paths.add(image["imagePath"])
        _sha(image["imageSha256"], "image.imageSha256")
        _require(type(image["candidates"]) is list and bool(image["candidates"]), "Image candidates must be nonempty")
        for candidate in image["candidates"]:
            _fields(candidate, {"id", "sourceAnnotationId", "structureId", "bboxXYWH", "source",
                                "mask", "maskSha256", "proposalScore", "scoreMeaning", "warnings"}, "candidate")
            _integer(candidate["sourceAnnotationId"], "sourceAnnotationId")
            _require(candidate["id"] == f"endoscapes-box-{candidate['sourceAnnotationId']}",
                     "Candidate ID must match source annotation ID")
            _require(candidate["id"] not in candidate_ids and candidate["sourceAnnotationId"] not in annotation_ids,
                     "Duplicate candidate/source annotation ID")
            candidate_ids.add(candidate["id"])
            annotation_ids.add(candidate["sourceAnnotationId"])
            _require(isinstance(candidate["structureId"], str) and candidate["structureId"] in class_ids,
                     "Candidate structure must belong to bundle classes")
            _require(candidate["source"] == "model_generated", "Proposals must remain model_generated")
            bbox = candidate["bboxXYWH"]
            _require(type(bbox) is list and len(bbox) == 4, "bboxXYWH must contain four numbers")
            for value in bbox:
                _number(value, "bboxXYWH coordinate", 0)
            x, y, width, height = bbox
            _require(width > 0 and height > 0 and x + width <= image["width"] and y + height <= image["height"],
                     "Source box must have positive size within original image bounds")
            pixels = decode_rle(candidate["mask"])
            _require((candidate["mask"]["width"], candidate["mask"]["height"]) == (image["width"], image["height"]),
                     "Proposal dimensions must match image dimensions")
            _sha(candidate["maskSha256"], "candidate.maskSha256")
            _require(hashlib.sha256(pixels).hexdigest() == candidate["maskSha256"], "Proposal mask SHA-256 mismatch")
            if candidate["proposalScore"] is not None:
                _number(candidate["proposalScore"], "proposalScore")
            _text(candidate["scoreMeaning"], "scoreMeaning")
            _require(type(candidate["warnings"]) is list and all(isinstance(item, str) for item in candidate["warnings"]),
                     "warnings must be an array of strings")


def validate_review(review, bundle, bundle_sha256):
    """Return counts/eligible IDs after validating every supplied decision.

    Missing decisions and pending records are unreviewed. Only explicit anatomy
    scope accepted/edited decisions are eligible, with nonempty positive masks.
    This checks attribution and consistency, not the reviewer's qualifications.
    """
    validate_bundle(bundle)
    _sha(bundle_sha256, "Expected bundle SHA-256")
    _fields(review, {"formatVersion", "artifactType", "bundleId", "bundleSha256", "reviewer", "exportedAt", "decisions"}, "review")
    _require(review["formatVersion"] == VERSION and review["artifactType"] == "candidate_mask_review",
             "Unsupported review type/version")
    _require(review["bundleId"] == bundle["bundleId"] and review["bundleSha256"] == bundle_sha256,
             "Review bundle identity/SHA-256 mismatch")
    _fields(review["reviewer"], {"name", "reviewScope"}, "reviewer")
    _text(review["reviewer"]["name"], "reviewer.name")
    _require(review["reviewer"]["reviewScope"] in ("anatomy", "technical"), "Invalid reviewer scope")
    _utc(review["exportedAt"], "exportedAt")
    _require(type(review["decisions"]) is list, "decisions must be an array")
    lookup = {candidate["id"]: (image, candidate) for image in bundle["images"] for candidate in image["candidates"]}
    seen, eligible, counts = set(), [], Counter({key: 0 for key in sorted(DECISIONS)})
    for record in review["decisions"]:
        _fields(record, {"candidateId", "imageId", "imageSha256", "proposalMaskSha256", "decision",
                         "mask", "notes", "reviewedAt", "reviewMilliseconds"}, "decision")
        candidate_id = record["candidateId"]
        _require(isinstance(candidate_id, str) and candidate_id in lookup and candidate_id not in seen,
                 "Unknown or duplicate candidate decision")
        seen.add(candidate_id)
        image, candidate = lookup[candidate_id]
        _require(record["imageId"] == image["id"] and record["imageSha256"] == image["imageSha256"],
                 "Review image identity/SHA-256 mismatch")
        _require(record["proposalMaskSha256"] == candidate["maskSha256"], "Review proposal SHA-256 mismatch")
        decision = record["decision"]
        _require(isinstance(decision, str) and decision in DECISIONS, "Invalid decision state")
        _require(isinstance(record["notes"], str), "Decision notes must be text")
        _number(record["reviewMilliseconds"], "reviewMilliseconds", 0)
        pixels = decode_rle(record["mask"])
        _require((record["mask"]["width"], record["mask"]["height"]) == (image["width"], image["height"]),
                 "Review mask dimensions must match image dimensions")
        if decision == "pending":
            _require(record["reviewedAt"] is None, "Pending candidates must not carry a reviewedAt timestamp")
        else:
            _utc(record["reviewedAt"], "reviewedAt")
        if decision in {"rejected", "needs_expert"}:
            _require(bool(record["notes"].strip()), "Rejected/needs_expert decisions require a note")
        if decision in {"accepted", "edited"}:
            _require(any(pixels), "Accepted/edited masks must have foreground pixels")
            identical = pixels == decode_rle(candidate["mask"])
            _require(identical if decision == "accepted" else not identical,
                     "Accepted masks must equal proposal; edited masks must differ")
            if review["reviewer"]["reviewScope"] == "anatomy":
                eligible.append(candidate_id)
        counts[decision] += 1
    return {
        "candidateCount": len(lookup), "suppliedDecisionCount": len(seen),
        "missingDecisionCount": len(lookup) - len(seen), "decisionCounts": dict(counts),
        "eligibleCandidateIds": eligible,
        "unreviewedCandidateCount": len(lookup) - len(seen) + counts["pending"],
    }


def _decode_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid_constant), hashlib.sha256(raw).hexdigest()


def _read_json(path):
    return _decode_json(Path(path).read_bytes())


def validate_bundle_files(bundle, bundle_path):
    """Verify all local source image bytes/dimensions, disallow path escapes."""
    from PIL import Image

    validate_bundle(bundle)
    base = Path(bundle_path).resolve().parent
    for image in bundle["images"]:
        path = (base / image["imagePath"]).resolve()
        _require(path.is_relative_to(base), "Image path escapes bundle directory (including symlinks)")
        _require(path.is_file(), f"Missing bundle image: {image['imagePath']}")
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == image["imageSha256"], f"Image SHA-256 mismatch: {image['id']}")
        with Image.open(io.BytesIO(raw)) as source:
            _require(source.size == (image["width"], image["height"]), f"Image dimensions mismatch: {image['id']}")
            source.verify()


def import_review(bundle_path, review_path, output_dir):
    """Validate everything first, then publish a new receipt/mask directory.

    Existing destinations are never overwritten. No output is created for an
    invalid record or stale image. Eligible masks remain independent even when
    different candidate classes overlap; their zeros mean unknown supervision.
    """
    from PIL import Image

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite review import: {destination}")
    bundle, bundle_digest = _read_json(bundle_path)
    review_raw = Path(review_path).read_bytes()
    review, review_digest = _decode_json(review_raw)
    summary = validate_review(review, bundle, bundle_digest)
    validate_bundle_files(bundle, bundle_path)
    lookup = {candidate["id"]: (image, candidate) for image in bundle["images"] for candidate in image["candidates"]}
    decisions = {record["candidateId"]: record for record in review["decisions"]}
    receipt = {
        "formatVersion": VERSION, "artifactType": "candidate_mask_review_receipt",
        "bundleId": bundle["bundleId"], "bundleSha256": bundle_digest,
        "sourceBundlePath": str(Path(bundle_path).resolve()),
        "reviewSha256": review_digest, "reviewer": review["reviewer"],
        "exportedAt": review["exportedAt"], "importedAt": datetime.now(timezone.utc).isoformat(),
        "summary": summary, "eligibleMasks": [], "trainingEnrollment": "none",
        "maskSemantics": {"255": "reviewed candidate foreground", "0": "unknown; not a background or negative label"},
        "coverage": "partial positive masks; candidates may overlap and are never merged into a dense target",
    }
    # All external data and identities have passed validation before any mkdir
    # or PNG write. Stage complete output, reserve an exclusive destination,
    # then publish the directory. Cleanup never recursively deletes destination.
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    reserved = False
    try:
        (stage / "masks").mkdir()
        for candidate_id in summary["eligibleCandidateIds"]:
            image, candidate = lookup[candidate_id]
            decision = decisions[candidate_id]
            pixels = decode_rle(decision["mask"])
            relative = f"masks/{candidate_id}.png"
            png = Image.frombytes("L", (image["width"], image["height"]), pixels.translate(bytes([0, 255] + list(range(2, 256)))))
            png.save(stage / relative)
            metadata = {
                "candidateId": candidate_id, "imageId": image["id"], "imageSha256": image["imageSha256"],
                "imagePath": image["imagePath"], "imagePathRelativeTo": "source bundle directory",
                "videoId": image["videoId"], "frameNumber": image["frameNumber"],
                "split": "train", "width": image["width"], "height": image["height"],
                "structureId": candidate["structureId"], "sourceAnnotationId": candidate["sourceAnnotationId"],
                "bboxXYWH": candidate["bboxXYWH"], "proposalMaskSha256": candidate["maskSha256"],
                "reviewedMaskSha256": hashlib.sha256(pixels).hexdigest(), "maskPath": relative,
                "maskFileSha256": hashlib.sha256((stage / relative).read_bytes()).hexdigest(),
                "decision": decision["decision"], "reviewer": review["reviewer"],
                "notes": decision["notes"], "reviewedAt": decision["reviewedAt"],
                "reviewMilliseconds": decision["reviewMilliseconds"],
                "maskSemantics": receipt["maskSemantics"], "coverage": "partial_positive_only",
            }
            (stage / f"masks/{candidate_id}.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            receipt["eligibleMasks"].append(metadata)
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        # Preserve every decision, including pending and excluded decisions, so
        # the receipt is auditable without converting those states to labels.
        (stage / "review.json").write_bytes(review_raw)
        destination.mkdir()  # Exclusive even if another import raced with us.
        reserved = True
        stage.rename(destination)
        reserved = False
    except BaseException:
        if reserved:
            destination.rmdir()
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="Validate bundle/source files and optionally a review")
    check.add_argument("--bundle", type=Path, required=True)
    check.add_argument("--review", type=Path)
    ingest = commands.add_parser("import-review", help="Import eligible partial masks into a new directory")
    ingest.add_argument("--bundle", type=Path, required=True)
    ingest.add_argument("--review", type=Path, required=True)
    ingest.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "import-review":
        result = import_review(args.bundle, args.review, args.output_dir)
    else:
        bundle, digest = _read_json(args.bundle)
        validate_bundle_files(bundle, args.bundle)
        result = {"valid": True, "bundleId": bundle["bundleId"], "bundleSha256": digest}
        if args.review:
            review, _ = _read_json(args.review)
            result["review"] = validate_review(review, bundle, digest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
