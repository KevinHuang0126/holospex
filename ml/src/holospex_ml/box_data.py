"""Acquire only the additional TRAIN cases of public Endoscapes-BBox201.

Bounding boxes are retained as boxes. This module never fabricates pixel masks,
modifies the Seg50 manifest, or downloads held-out imagery. Network acquisition
is explicit; importing this module has no side effects.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import threading
import zipfile
import zlib

from PIL import Image

from .dataset import _structure_id
from .download import (
    DownloadError, HTTPRangeReader, MAX_OUTPUT_BYTES, OFFICIAL_ARCHIVE_URL,
    OFFICIAL_RELEASE_PAGE, _destination, _read_member, _read_member_independent,
    _write_verified, resolve_member,
)

METADATA = ("LICENSE", "README.md", "train_vids.txt", "val_vids.txt", "test_vids.txt",
            "train_seg_vids.txt", "val_seg_vids.txt", "test_seg_vids.txt",
            "train/annotation_coco.json")


def encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_content(image: dict, data: bytes, existing_hashes: dict) -> dict:
    """Keep duplicated or blank raw frames out of the usable training manifest."""
    with Image.open(io.BytesIO(data)) as decoded:
        if decoded.format != "JPEG" or decoded.size != (image["width"], image["height"]):
            raise DownloadError("Downloaded image type/dimensions disagree with source metadata")
        decoded.load()
        black = all(bounds == (0, 0) for bounds in decoded.convert("RGB").getextrema())
    digest = hashlib.sha256(data).hexdigest()
    matches = existing_hashes.get(digest, [])
    reasons = []
    if matches:
        reasons.append("exact_content_match_to_existing_seg50")
    if black:
        reasons.append("uniform_black_frame")
    return {"sha256": digest, "bytes": len(data), "uniformBlackFrame": black,
            "existingSeg50Matches": matches, "quarantineReasons": reasons,
            "acquisitionStatus": "quarantined" if reasons else "usable"}


def read_cases(data: str) -> set[int]:
    numbers = [float(value) for value in data.split()]
    if not numbers or any(not math.isfinite(v) or v < 1 or not v.is_integer() for v in numbers):
        raise DownloadError("Video lists must contain positive integer case IDs")
    values = [int(v) for v in numbers]
    if len(values) != len(set(values)):
        raise DownloadError("Duplicate case IDs in video list")
    return set(values)


def image_identity(image: dict) -> tuple[str, int, int]:
    filename = image.get("file_name", "")
    match = re.fullmatch(r"([0-9]+)_([0-9]+)\.jpg", filename)
    if not match:
        raise DownloadError(f"Unexpected frame filename: {filename!r}")
    video, frame = map(int, match.groups())
    if image.get("video_id") != video:
        raise DownloadError(f"Filename and annotation video disagree: {filename}")
    for field in ("id", "width", "height"):
        value = image.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DownloadError(f"Invalid image {field}: {filename}")
    return filename, video, frame


def plan_expansion(coco: dict, segmentation: dict, cases: dict, seg_cases: dict) -> dict:
    """Validate source identity and splits before selecting any image transfer."""
    for name, groups in (("full", cases), ("Seg50", seg_cases)):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if groups[a] & groups[b]:
                raise DownloadError(f"Overlapping {name} {a}/{b} cases")
    if any(not seg_cases[s] <= cases[s] for s in ("train", "val", "test")):
        raise DownloadError("Seg50 split cases do not align with official full splits")
    heldout = cases["val"] | cases["test"] | seg_cases["val"] | seg_cases["test"]
    categories = {item["id"]: {**item, "structureId": _structure_id(item["name"])}
                  for item in coco["categories"]}
    if len(categories) != len(coco["categories"]) or len(set(c["structureId"] for c in categories.values())) != 6:
        raise DownloadError("Require six distinct canonical anatomy/tool categories")
    if {c["id"]: c["name"] for c in coco["categories"]} != {c["id"]: c["name"] for c in segmentation["categories"]}:
        raise DownloadError("Bounding-box and Seg50 category maps disagree")
    seg_names = {image_identity(image)[0] for image in segmentation["images"]}
    if len(seg_names) != len(segmentation["images"]):
        raise DownloadError("Duplicate Seg50 frame filename")
    seg_image_cases = {image_identity(image)[1] for image in segmentation["images"]}
    if seg_image_cases != seg_cases["train"]:
        raise DownloadError("Seg50 training images disagree with its case list")
    image_by_id, filename_set, selected = {}, set(), []
    for image in coco["images"]:
        filename, video, frame = image_identity(image)
        if image["id"] in image_by_id or filename in filename_set:
            raise DownloadError("Duplicate source image ID or filename")
        image_by_id[image["id"]] = image
        filename_set.add(filename)
        if video not in cases["train"] or video in heldout:
            raise DownloadError(f"Training annotation references a held-out/nontraining case: {filename}")
        if filename in seg_names:
            continue
        if video in seg_cases["train"]:
            raise DownloadError("Expansion unexpectedly contains extra frames from an existing Seg50 case")
        selected.append({"imageId": image["id"], "fileName": filename, "imagePath": f"images/{filename}",
                         "videoId": video, "frameNumber": frame,
                         "width": image["width"], "height": image["height"],
                         "split": "train", "annotationType": "bounding_box", "boxes": []})
    if not seg_names <= filename_set:
        raise DownloadError("Seg50 train is not a strict subset of BBox201 train")
    selected_by_id = {image["imageId"]: image for image in selected}
    annotation_ids = set()
    for annotation in coco["annotations"]:
        if annotation["id"] in annotation_ids or annotation["image_id"] not in image_by_id:
            raise DownloadError("Duplicate annotation ID or missing referenced image")
        annotation_ids.add(annotation["id"])
        if annotation["image_id"] not in selected_by_id:
            continue
        image = selected_by_id[annotation["image_id"]]
        box = annotation.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            raise DownloadError("Bounding boxes must contain four finite numeric xywh values")
        x, y, width, height = box
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image["width"] or y + height > image["height"]:
            raise DownloadError(f"Invalid/out-of-image bounding box: {image['fileName']}")
        if annotation.get("segmentation"):
            raise DownloadError("Unexpected segmentation payload in box-only expansion; inspect release before proceeding")
        if annotation["category_id"] not in categories:
            raise DownloadError("Unknown source box category")
        image["boxes"].append({"annotationId": annotation["id"], "sourceCategoryId": annotation["category_id"],
                               "structureId": categories[annotation["category_id"]]["structureId"],
                               "bboxXYWH": box, "iscrowd": annotation.get("iscrowd", 0),
                               "sourceArea": annotation.get("area")})
    for image in selected:
        image["hasBoxAnnotations"] = bool(image["boxes"])
        # Empty annotations do not establish an all-background pixel mask.
        image["labelCompleteness"] = "not_established"
    selected.sort(key=lambda image: (image["videoId"], image["frameNumber"]))
    selected_cases = {image["videoId"] for image in selected}
    per_class = Counter(box["structureId"] for image in selected for box in image["boxes"])
    return {"schemaVersion": "1.0.0", "dataset": "Endoscapes-BBox201 train expansion",
            "annotationType": "bounding_box", "annotationSource": "author_dataset_annotation",
            "coordinateSpace": "original_pixels", "bboxFormat": "xywh", "split": "train",
            "pixelMasksAvailable": False, "pixelSupervision": "none", "labelCompleteness": "not_established",
            "categories": list(categories.values()), "images": selected,
            "summary": {"bboxTrainImageCount": len(image_by_id), "existingSeg50TrainImages": len(seg_names),
                        "additionalImages": len(selected), "additionalCases": len(selected_cases),
                        "boxAnnotations": sum(per_class.values()), "boxesByStructure": dict(sorted(per_class.items())),
                        "framesWithBoxes": sum(bool(image["boxes"]) for image in selected),
                        "framesWithoutBoxes": sum(not image["boxes"] for image in selected),
                        "emptyAnnotationFrames": [image["fileName"] for image in selected if not image["boxes"]]},
            "leakageAudit": {"officialTrainCases": sorted(cases["train"]), "selectedTrainCases": sorted(selected_cases),
                             "existingSeg50TrainCases": sorted(seg_cases["train"]),
                             "heldOutCases": sorted(heldout), "heldOutCaseOverlap": [],
                             "existingSeg50CaseOverlap": [], "existingSeg50FrameOverlap": [],
                             "newHeldOutImageDownloads": 0}}


def acquire_expansion(seg50_root: Path, output_dir: Path, *, download_images=False, progress=print) -> dict:
    root, seg_root = Path(output_dir).resolve(), Path(seg50_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries, metadata = [], {}
    progress("Reading the official ZIP index and training metadata; no held-out images are selected")
    with HTTPRangeReader(OFFICIAL_ARCHIVE_URL) as reader, zipfile.ZipFile(reader) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise DownloadError("Archive contains duplicate member names")
        link_cache = {}

        def record(relative, info, data, transferred):
            entries.append({"path": relative, "archivePath": info.filename, "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(), "crc32": f"{info.CRC:08x}",
                            "downloadedThisRun": transferred})

        def cached(relative, info):
            path = _destination(root, relative)
            if path.is_file() and path.stat().st_size == info.file_size:
                data = path.read_bytes()
                if zlib.crc32(data) == info.CRC:
                    return data
            return None

        for name in METADATA:
            info = resolve_member(archive, "endoscapes/" + name, link_cache=link_cache)
            relative = "source/" + name
            data = cached(relative, info)
            downloaded = data is None
            if downloaded:
                data = _read_member(archive, info)
                _write_verified(root, relative, data)
            metadata[name] = data
            record(relative, info, data, downloaded)
        cases = {s: read_cases(metadata[s + "_vids.txt"].decode()) for s in ("train", "val", "test")}
        seg_cases = {s: read_cases(metadata[s + "_seg_vids.txt"].decode()) for s in ("train", "val", "test")}
        # Compare every official case list with the already acquired Seg50
        # release, not just one train/val threshold hard-coded from filenames.
        for s in ("train", "val", "test"):
            for suffix in ("_vids.txt", "_seg_vids.txt"):
                if read_cases((seg_root / (s + suffix)).read_text()) != read_cases(metadata[s + suffix].decode()):
                    raise DownloadError("Current Seg50 and remote release split lists disagree")
        seg_coco_path = seg_root / "train_seg/annotation_coco.json"
        plan = plan_expansion(json.loads(metadata["train/annotation_coco.json"]),
                              json.loads(seg_coco_path.read_text()), cases, seg_cases)
        expected = {"bboxTrainImageCount": 1212, "existingSeg50TrainImages": 343,
                    "additionalImages": 869, "additionalCases": 90, "boxAnnotations": 3951}
        if any(plan["summary"][key] != value for key, value in expected.items()):
            raise DownloadError(f"Release counts changed; inspect before bulk transfer: {plan['summary']}")
        _write_verified(root, "plan.json", encoded(plan))
        progress("Verified expansion: 869 new training frames, 90 cases, 3,951 bounding boxes")
        workers, local, worker_lock = [], threading.local(), threading.Lock()
        if download_images:
            infos = {image["fileName"]: resolve_member(archive, "endoscapes/all/" + image["fileName"], link_cache=link_cache)
                     for image in plan["images"]}
            if sum(info.file_size for info in infos.values()) + sum(len(v) for v in metadata.values()) > MAX_OUTPUT_BYTES:
                raise DownloadError("Selected expansion exceeds the bounded output budget")
            existing_hashes = defaultdict(list)
            for split in ("train", "val", "test"):
                for path in (seg_root / (split + "_seg")).glob("*.jpg"):
                    existing_hashes[file_sha256(path)].append({"split": split, "fileName": path.name})

            def fetch(image):
                info, relative = infos[image["fileName"]], image["imagePath"]
                data = cached(relative, info)
                if data is None:
                    data = cached("quarantine/" + relative, info)
                downloaded = data is None
                if downloaded:
                    if not hasattr(local, "reader"):
                        local.reader = HTTPRangeReader(OFFICIAL_ARCHIVE_URL, metadata=(reader.size, reader.etag), budget=reader._budget)
                        with worker_lock:
                            workers.append(local.reader)
                    data = _read_member_independent(local.reader, info)
                content = inspect_content(image, data, existing_hashes)
                if content["quarantineReasons"]:
                    relative = "quarantine/images/" + image["fileName"]
                    # A resumed run might already contain a verified raw file
                    # under images/. Its existence never makes it eligible:
                    # the manifest below remains the authority for selection.
                if downloaded or content["quarantineReasons"]:
                    _write_verified(root, relative, data)
                image.update(content)
                image["imagePath"] = relative
                return image, info, data, downloaded

            ordered = sorted(plan["images"], key=lambda image: infos[image["fileName"]].header_offset)
            try:
                with ThreadPoolExecutor(max_workers=6) as executor:
                    for number, (image, info, data, downloaded) in enumerate(executor.map(fetch, ordered), 1):
                        record(image["imagePath"], info, data, downloaded)
                        if number % 50 == 0 or number == len(ordered):
                            progress(f"Training JPEGs verified: {number}/{len(ordered)}")
            finally:
                for worker in workers:
                    worker.close()
            quarantined = [image for image in plan["images"] if image["acquisitionStatus"] == "quarantined"]
            preexisting_duplicates = [{"sha256": digest, "frames": matches} for digest, matches in existing_hashes.items()
                                      if len(matches) > 1]
            plan["leakageAudit"].update({"usableExistingSeg50ImageSHA256Overlap": [],
                                        "existingSeg50ImagesHashChecked": sum(len(v) for v in existing_hashes.values()),
                                        "preExistingSeg50DuplicateGroups": preexisting_duplicates,
                                        "quarantinedFrames": [{k: image[k] for k in (
                                            "fileName", "videoId", "sha256", "imagePath", "quarantineReasons",
                                            "existingSeg50Matches", "uniformBlackFrame", "boxes")} for image in quarantined]})
            _write_verified(root, "acquired-manifest.json", encoded(plan))
            usable = {**plan, "images": [image for image in plan["images"] if image["acquisitionStatus"] == "usable"]}
            usable_counts = Counter(box["structureId"] for image in usable["images"] for box in image["boxes"])
            usable["summary"] = {**plan["summary"], "sourceAdditionalImages": len(plan["images"]),
                                 "sourceAdditionalCases": plan["summary"]["additionalCases"],
                                 "quarantinedImages": len(quarantined), "additionalImages": len(usable["images"]),
                                 "additionalCases": len({image["videoId"] for image in usable["images"]}),
                                 "boxAnnotations": sum(usable_counts.values()), "boxesByStructure": dict(sorted(usable_counts.items())),
                                 "framesWithBoxes": sum(bool(image["boxes"]) for image in usable["images"]),
                                 "framesWithoutBoxes": sum(not image["boxes"] for image in usable["images"]),
                                 "emptyAnnotationFrames": [image["fileName"] for image in usable["images"] if not image["boxes"]]}
            _write_verified(root, "manifest.json", encoded(usable))
            progress(f"Wrote usable box manifest: {len(usable['images'])} images, {len(quarantined)} quarantined")
        entries.sort(key=lambda entry: entry["path"])
        report = {"sourceUrl": OFFICIAL_ARCHIVE_URL, "sourcePage": OFFICIAL_RELEASE_PAGE,
                  "archiveBytes": reader.size, "archiveEtag": reader.etag,
                  "downloadedAt": datetime.now(timezone.utc).isoformat(), "metadataOnly": not download_images,
                  "annotationType": "bounding_box", "subset": "additional official BBox201 training cases only",
                  "license": "CC BY-NC-SA 4.0; non-commercial scientific research; see source/LICENSE",
                  "transferredBytes": reader.transferred_bytes + sum(w.transferred_bytes for w in workers),
                  "selectedBytes": sum(item["bytes"] for item in entries), "selectedEntries": entries,
                  "seg50TrainAnnotationsSha256": file_sha256(seg_coco_path), "summary": plan["summary"],
                  "usableSummary": usable["summary"] if download_images else None,
                  "leakageAudit": plan["leakageAudit"],
                  "selectedEntriesSha256": hashlib.sha256(encoded(entries)).hexdigest()}
    report_name = "download-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    _write_verified(root, report_name, encoded(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg50-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Explicitly download the 869 audited training JPEGs")
    args = parser.parse_args(argv)
    report = acquire_expansion(args.seg50_root, args.output_dir, download_images=args.download)
    print(json.dumps({k: report[k] for k in ("summary", "metadataOnly", "transferredBytes", "selectedBytes")}, indent=2))


if __name__ == "__main__":
    main()
