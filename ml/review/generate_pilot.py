"""Freeze train-only cases, then generate immutable box-prompted review proposals.

Preparation needs only the standard library. SAM/PyTorch imports happen solely
in ``generate`` on the cloud GPU. No proposal is accepted anatomy supervision.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import tempfile
import time

SMALL_CLASSES = ("cystic_artery", "cystic_plate", "cystic_duct",
                 "hepatocystic_triangle_dissection")
QUOTAS = (8, 8, 2, 2)
SCORE_MEANING = "SAM predicted mask overlap quality; not anatomy confidence or clinical correctness."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or ".." in p.parts or "\\" in relative:
        raise ValueError("Expected a safe relative file path")
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes its input root")
    return target


def box_geometry(box: dict, others: list[dict]) -> dict:
    x, y, w, h = box["bboxXYWH"]
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("Invalid source box")
    overlap = 0.0
    for other in others:
        if other["annotationId"] == box["annotationId"]:
            continue
        ox, oy, ow, oh = other["bboxXYWH"]
        overlap = max(overlap, max(0, min(x+w, ox+ow)-max(x, ox)) *
                      max(0, min(y+h, oy+oh)-max(y, oy)) / (w*h))
    aspect = max(w/h, h/w)
    return {"boxAreaPixels": w*h, "aspectRatio": aspect,
            "maximumOtherSmallBoxOverlapFraction": overlap,
            "geometryDifficultyProxy": "higher" if aspect >= 3 or overlap >= .25 else "lower"}


def select_images(manifest: dict, seed: int = 42, *, image_count: int = 20,
                  case_count: int = 20, excluded_cases=(), min_frame_gap: int = 750) -> tuple[list[dict], dict]:
    """Source-box strata only; use one or two temporally separated images per case.

    The first image is an artery/plate-weighted case anchor. For a second image,
    prefer that same class in a different box-area/difficulty stratum. This is a
    reproducible sampling policy, not a claim that either image contains an error.
    """
    if (type(image_count) is not int or type(case_count) is not int or case_count < 4
            or image_count not in (case_count, 2 * case_count)):
        raise ValueError("Choose at least four cases and exactly one or two images per case")
    if type(min_frame_gap) is not int or min_frame_gap < 1:
        raise ValueError("Minimum frame gap must be a positive integer")
    per_case = image_count // case_count
    if manifest.get("split") != "train" or manifest.get("pixelMasksAvailable") is not False:
        raise ValueError("Expected the official train-only box expansion manifest")
    audit = manifest["leakageAudit"]
    protected = set(map(str, audit["heldOutCases"])) | set(map(str, audit["existingSeg50TrainCases"]))
    allowed = set(map(str, audit["officialTrainCases"]))
    excluded = set(map(str, excluded_cases))
    pools = {key: [] for key in SMALL_CLASSES}
    for image in sorted(manifest["images"], key=lambda i: i["fileName"]):
        case = str(image["videoId"])
        if image["split"] != "train" or case not in allowed or case in protected:
            raise ValueError("Expansion contains a protected or non-train case")
        if (image["acquisitionStatus"] != "usable" or image.get("quarantineReasons")
                or case in excluded):
            continue
        boxes = [b for b in image["boxes"] if b["structureId"] in SMALL_CLASSES]
        for box in boxes:
            geometry = box_geometry(box, boxes)
            x, y, w, h = box["bboxXYWH"]
            if x < 0 or y < 0 or x+w > image["width"] or y+h > image["height"]:
                raise ValueError("Source box exceeds image bounds")
            rank = hashlib.sha256(f"{seed}:{box['annotationId']}".encode()).hexdigest()
            pools[box["structureId"]].append((image, box, geometry, rank))
    thresholds = {}
    for key, pool in pools.items():
        values = sorted(row[2]["boxAreaPixels"] for row in pool)
        if not values:
            raise ValueError(f"No available boxes for {key}")
        thresholds[key] = [values[len(values)//3], values[2*len(values)//3]]
        for _, _, geometry, _ in pool:
            geometry["areaTercile"] = sum(geometry["boxAreaPixels"] > edge for edge in thresholds[key])
    # Largest-remainder allocation preserves 8/8/2/2 for the original 20 cases.
    quotas = [case_count * weight // sum(QUOTAS) for weight in QUOTAS]
    remainder_order = sorted(range(len(QUOTAS)),
                             key=lambda i: (-(case_count * QUOTAS[i] % sum(QUOTAS)), i))
    for index in remainder_order[:case_count - sum(quotas)]:
        quotas[index] += 1
    by_case = {}
    for pool in pools.values():
        for row in pool:
            by_case.setdefault(str(row[0]["videoId"]), []).append(row)

    def companions(row):
        return [other for other in by_case[str(row[0]["videoId"])]
                if other[0]["fileName"] != row[0]["fileName"]
                and abs(other[0]["frameNumber"] - row[0]["frameNumber"]) >= min_frame_gap]

    selected, anchors, cases = [], [], set()

    def add(row, key, target_bin, target_difficulty, role):
        image, box, geometry, _ = row
        selected.append(image)
        anchors.append({"imageId": Path(image["fileName"]).stem,
                        "videoId": str(image["videoId"]), "role": role,
                        "sourceAnnotationId": box["annotationId"], "structureId": box["structureId"],
                        "requestedStructureId": key, "requestedAreaTercile": target_bin,
                        "requestedDifficultyProxy": target_difficulty, **geometry})

    gaps = []
    for key, quota in zip(SMALL_CLASSES, quotas):
        for slot in range(quota):
            target_bin = slot % 3
            target_difficulty = "higher" if slot % 2 else "lower"
            pool = [r for r in pools[key] if str(r[0]["videoId"]) not in cases
                    and (per_case == 1 or companions(r))]
            if not pool:
                raise ValueError("Insufficient distinct eligible cases for frozen quotas and frame gap")
            row = min(pool, key=lambda r: (r[2]["areaTercile"] != target_bin,
                      r[2]["geometryDifficultyProxy"] != target_difficulty, r[3]))
            cases.add(str(row[0]["videoId"]))
            add(row, key, target_bin, target_difficulty, "case_anchor")
            if per_case == 2:
                second_bin = (row[2]["areaTercile"] + 1) % 3
                second_difficulty = "lower" if row[2]["geometryDifficultyProxy"] == "higher" else "higher"
                second = min(companions(row), key=lambda r: (r[1]["structureId"] != key,
                             r[2]["areaTercile"] != second_bin,
                             r[2]["geometryDifficultyProxy"] != second_difficulty, r[3]))
                add(second, key, second_bin, second_difficulty, "case_companion")
                gaps.append({"videoId": str(row[0]["videoId"]),
                             "frameGap": abs(second[0]["frameNumber"] - row[0]["frameNumber"])})
    selection = {"seed": seed, "method": "fixed case class quotas; source box-area terciles; geometry strata; SHA256(seed:annotationId) tie order",
                 "anchorQuotas": dict(zip(SMALL_CLASSES, quotas)),
                 "imageAnchorCountsByClass": dict(Counter(a["structureId"] for a in anchors)),
                 "areaTercileUpperEdgesPixels": thresholds, "anchors": anchors,
                 "difficultyMeaning": "Geometry proxies only: aspect ratio >=3 or overlap with another small-class box >=0.25. Not assessed clinical difficulty.",
                 "fallbackPolicy": "Case anchor: prefer requested area tercile, then difficulty in an unused case. Companion: require frame gap, then prefer same class, requested area tercile, then difficulty. Record every actual anchor.",
                 "imageCount": len(selected), "distinctCaseCount": len(cases),
                 "imagesPerCase": per_case, "minimumFrameGap": min_frame_gap if per_case == 2 else None,
                 "caseFrameGaps": gaps, "excludedPriorReviewCases": sorted(excluded),
                 "allSmallClassBoxesIncluded": True, "modelScoresUsedForSelection": False,
                 "split": "train", "protectedCasesExcluded": sorted(protected)}
    return selected, selection


def prepare(manifest_path: Path, output: Path, anatomy_path: Path, seed: int = 42, *,
            image_count: int = 20, case_count: int = 20, exclude_bundles=(),
            bundle_id: str = "review-pilot-001", min_frame_gap: int = 750) -> None:
    if output.exists():
        raise ValueError("Input bundle directory must be new")
    if not bundle_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in bundle_id):
        raise ValueError("Bundle ID must use lowercase letters, digits, hyphens, or underscores")
    manifest = json.loads(manifest_path.read_text())
    excluded_cases, exclusion_records = set(), []
    for index, bundle_path in enumerate(exclude_bundles):
        bundle_path = Path(bundle_path)
        bundle = json.loads(bundle_path.read_text())
        if (bundle.get("artifactType") != "candidate_mask_bundle" or not bundle.get("images")
                or any(i.get("split") != "train" for i in bundle["images"])):
            raise ValueError("Excluded bundle must contain train-only candidate images")
        cases = sorted({str(i["videoId"]) for i in bundle["images"]})
        excluded_cases.update(cases)
        exclusion_records.append({"bundleId": bundle["bundleId"], "sha256": sha256(bundle_path),
                                  "imageCount": len(bundle["images"]), "cases": cases,
                                  "snapshotPath": f"source/excluded-bundle-{index + 1:03d}.json"})
    selected, selection = select_images(manifest, seed, image_count=image_count,
                                       case_count=case_count, excluded_cases=excluded_cases,
                                       min_frame_gap=min_frame_gap)
    selection["excludedPriorBundles"] = exclusion_records
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete frozen input, so failed copying cannot look ready.
    with tempfile.TemporaryDirectory(prefix=output.name + "-", dir=output.parent) as temporary:
        staging = Path(temporary) / "prepared"
        (staging / "images").mkdir(parents=True)
        images = []
        for source in selected:
            image = dict(source)
            path = safe_path(manifest_path.parent, source["imagePath"])
            if sha256(path) != source["sha256"]:
                raise ValueError("Source image checksum mismatch")
            image["imagePath"] = "images/" + source["fileName"]
            image["boxes"] = [b for b in source["boxes"] if b["structureId"] in SMALL_CLASSES]
            shutil.copyfile(path, safe_path(staging, image["imagePath"]))
            images.append(image)
        source_dir = staging / "source"
        source_dir.mkdir()
        shutil.copyfile(manifest_path, source_dir / "manifest.json")
        for name in ("LICENSE", "README.md"):
            shutil.copyfile(manifest_path.parent / "source" / name, source_dir / name)
        for path, record in zip(exclude_bundles, exclusion_records):
            shutil.copyfile(path, safe_path(staging, record["snapshotPath"]))
        anatomy = json.loads(anatomy_path.read_text())
        selection["proposalCountsByClass"] = dict(Counter(b["structureId"] for i in images for b in i["boxes"]))
        frozen = {"bundleId": bundle_id, "sourceManifestSha256": sha256(manifest_path),
                  "selection": selection, "images": images,
                  "classes": [{"structureId": c, **anatomy[c]} for c in SMALL_CLASSES]}
        (staging / "selection.json").write_text(json.dumps(frozen, indent=2) + "\n")
        validate_frozen_selection(frozen, staging)
        staging.rename(output)
    print(json.dumps(selection, indent=2), flush=True)


def validate_frozen_selection(frozen: dict, input_dir: Path) -> None:
    """Check provenance and case boundaries before loading SAM or allocating GPU."""
    manifest_path = input_dir / "source/manifest.json"
    if sha256(manifest_path) != frozen["sourceManifestSha256"]:
        raise ValueError("Source manifest checksum mismatch")
    source = json.loads(manifest_path.read_text())
    selection, images = frozen["selection"], frozen["images"]
    cases = Counter(str(i["videoId"]) for i in images)
    if (len(images) != selection["imageCount"] or len(cases) != selection["distinctCaseCount"]
            or len({i["fileName"] for i in images}) != len(images)
            or set(cases.values()) != {selection.get("imagesPerCase", 1)}):
        raise ValueError("Frozen image or case counts mismatch")
    audit = source["leakageAudit"]
    protected = set(map(str, audit["heldOutCases"])) | set(map(str, audit["existingSeg50TrainCases"]))
    if set(cases) & protected or not set(cases) <= set(map(str, audit["officialTrainCases"])):
        raise ValueError("Frozen selection contains protected or non-train cases")
    excluded = set()
    for record in selection.get("excludedPriorBundles", []):
        path = safe_path(input_dir, record["snapshotPath"])
        if sha256(path) != record["sha256"]:
            raise ValueError("Excluded review bundle checksum mismatch")
        bundle = json.loads(path.read_text())
        actual = {str(i["videoId"]) for i in bundle["images"]}
        if actual != set(record["cases"]) or bundle["bundleId"] != record["bundleId"]:
            raise ValueError("Excluded review bundle case provenance mismatch")
        excluded.update(actual)
    if excluded != set(selection.get("excludedPriorReviewCases", [])) or set(cases) & excluded:
        raise ValueError("Frozen selection overlaps prior review cases")
    source_images = {i["fileName"]: i for i in source["images"]}
    for image in images:
        original = source_images[image["fileName"]]
        expected_boxes = [b for b in original["boxes"] if b["structureId"] in SMALL_CLASSES]
        if (image["boxes"] != expected_boxes or any(image[key] != original[key] for key in
                ("videoId", "frameNumber", "split", "sha256", "width", "height", "acquisitionStatus"))
                or image["split"] != "train" or image["acquisitionStatus"] != "usable"
                or image.get("quarantineReasons") or not expected_boxes):
            raise ValueError("Frozen image or boxes differ from eligible source")
        if sha256(safe_path(input_dir, image["imagePath"])) != image["sha256"]:
            raise ValueError("Frozen image checksum mismatch")
    if selection.get("imagesPerCase", 1) == 2:
        for case in cases:
            frames = [i["frameNumber"] for i in images if str(i["videoId"]) == case]
            if abs(frames[1] - frames[0]) < selection["minimumFrameGap"]:
                raise ValueError("Frozen same-case images violate minimum frame gap")


def encode_mask(mask) -> tuple[dict, str]:
    import numpy as np
    pixels = np.ascontiguousarray(mask, dtype=np.uint8)
    if pixels.ndim != 2 or not np.isin(pixels, [0, 1]).all():
        raise ValueError("Expected an unmodified two-dimensional binary mask")
    flat = pixels.reshape(-1)
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], changes, [len(flat)]))).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"encoding": "rle-row-major-v1", "width": pixels.shape[1],
            "height": pixels.shape[0], "counts": counts}, hashlib.sha256(pixels.tobytes()).hexdigest()


def mask_warnings(mask, box: list[float]) -> list[str]:
    import numpy as np
    x, y, w, h = box
    yy, xx = np.indices(mask.shape)
    outside = ~((xx+.5 >= x) & (xx+.5 < x+w) & (yy+.5 >= y) & (yy+.5 < y+h))
    area = int(mask.sum())
    warnings = []
    if not area:
        warnings.append("empty_mask")
    if np.any(mask & outside):
        warnings.append("mask_extends_outside_source_box")
    if area >= .25*mask.size or area > 4*w*h:
        warnings.append("large_mask_relative_to_image_or_box")
    return warnings


def generate(input_dir: Path, output: Path, checkpoint: Path, pin_path: Path, sam_repo: Path) -> None:
    frozen = json.loads((input_dir / "selection.json").read_text())
    validate_frozen_selection(frozen, input_dir)
    import numpy as np
    from PIL import Image
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if output.exists():
        raise ValueError("Output directory must be new")
    pin = json.loads(pin_path.read_text())
    if sha256(checkpoint) != pin["checkpointSha256"]:
        raise ValueError("Checkpoint does not match pinned SHA256")
    if subprocess.check_output(["git", "-C", str(sam_repo), "rev-parse", "HEAD"], text=True).strip() != pin["revision"]:
        raise ValueError("SAM source revision mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("This generator requires its authorized cloud CUDA GPU")
    torch.manual_seed(frozen["selection"]["seed"])
    np.random.seed(frozen["selection"]["seed"])
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    output.mkdir(parents=True)
    shutil.copytree(input_dir / "images", output / "images")
    shutil.copytree(input_dir / "source", output / "source")
    shutil.copyfile(input_dir / "selection.json", output / "selection.json")
    (output / "provenance").mkdir()
    shutil.copyfile(input_dir / "source/LICENSE", output / "provenance/Endoscapes_LICENSE.txt")
    shutil.copyfile(sam_repo / "LICENSE", output / "provenance/SAM2_LICENSE.txt")
    shutil.copyfile(pin_path, output / "provenance/sam-pin.json")
    (output / "masks").mkdir()
    predictor = SAM2ImagePredictor(build_sam2(pin["config"], str(checkpoint), device="cuda", apply_postprocessing=False),
                                  mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0)
    images, timings, ids = [], [], set()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for source in frozen["images"]:
            if source["split"] != "train" or source["acquisitionStatus"] != "usable":
                raise ValueError("Non-train or quarantined image in frozen selection")
            path = safe_path(input_dir, source["imagePath"])
            if sha256(path) != source["sha256"]:
                raise ValueError("Image checksum mismatch")
            rgb = Image.open(path).convert("RGB")
            if rgb.size != (source["width"], source["height"]):
                raise ValueError("Image dimensions mismatch")
            image = {"id": Path(source["fileName"]).stem, "videoId": str(source["videoId"]),
                     "frameNumber": source["frameNumber"], "split": "train", "width": rgb.width,
                     "height": rgb.height, "imagePath": source["imagePath"],
                     "imageSha256": source["sha256"], "candidates": []}
            torch.cuda.synchronize(); t0 = time.perf_counter()
            predictor.set_image(np.asarray(rgb))
            for box in source["boxes"]:
                if box["structureId"] not in SMALL_CLASSES:
                    raise ValueError("Unexpected proposal class")
                cid = f"endoscapes-box-{box['annotationId']}"
                if cid in ids:
                    raise ValueError("Duplicate candidate identity")
                ids.add(cid)
                x, y, w, h = box["bboxXYWH"]
                masks, scores, _ = predictor.predict(box=np.array([x, y, x+w, y+h], dtype=np.float32),
                                                     multimask_output=False, return_logits=False, normalize_coords=True)
                mask = np.asarray(masks[0], dtype=bool)
                if mask.shape != (rgb.height, rgb.width):
                    raise ValueError("Unexpected SAM mask dimensions")
                rle, digest = encode_mask(mask)
                score = float(np.asarray(scores).reshape(-1)[0])
                if not math.isfinite(score):
                    raise ValueError("Nonfinite SAM score")
                Image.fromarray(mask.astype(np.uint8)*255).save(output / "masks" / (cid + ".png"))
                image["candidates"].append({"id": cid, "sourceAnnotationId": box["annotationId"],
                    "structureId": box["structureId"], "bboxXYWH": box["bboxXYWH"],
                    "source": "model_generated", "mask": rle, "maskSha256": digest,
                    "proposalScore": score, "scoreMeaning": SCORE_MEANING,
                    "warnings": mask_warnings(mask, box["bboxXYWH"])})
            torch.cuda.synchronize()
            timings.append({"imageId": image["id"], "seconds": time.perf_counter()-t0,
                            "candidateCount": len(image["candidates"])})
            images.append(image)
            print(json.dumps({"image": image["id"], "completedImages": len(images), "candidates": len(ids)}), flush=True)
    generator = {**pin, "model": "SAM2.1 Hiera Large", "startedAt": started,
                 "completedAt": datetime.now(timezone.utc).isoformat(),
                 "runtimeSeconds": time.perf_counter()-begin, "perImageTimings": timings,
                 "device": torch.cuda.get_device_name(0), "cudaVersion": torch.version.cuda,
                 "python": platform.python_version(), "torch": torch.__version__,
                 "torchvision": importlib.metadata.version("torchvision"), "numpy": np.__version__,
                 "generatorScriptSha256": sha256(Path(__file__)),
                 "parameters": {"boxFormat": "original-image XYXY converted from unchanged source XYWH",
                    "multimask_output": False, "normalize_coords": True, "mask_threshold": 0.0,
                    "apply_postprocessing": False, "max_hole_area": 0.0, "max_sprinkle_area": 0.0,
                    "autocast": "bfloat16", "clipToBox": False, "removeComponents": False},
                 "warningMeaning": "Geometric review cues only, not correctness labels.",
                 "maskPngEncoding": "Additional PNG files store 0/255; maskSha256 hashes decoded row-major uint8 0/1.",
                 "trainingPerformed": False}
    bundle = {"formatVersion": "1.0.0", "artifactType": "candidate_mask_bundle",
              "bundleId": frozen.get("bundleId", "review-pilot-001"), "createdAt": generator["completedAt"],
              "sourceManifestSha256": frozen["sourceManifestSha256"], "selection": frozen["selection"],
              "generator": generator, "images": images, "classes": frozen["classes"]}
    (output / "bundle.json").write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "completed", "images": len(images), "candidates": len(ids),
                      "runtimeSeconds": generator["runtimeSeconds"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--anatomy", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--image-count", type=int, default=20)
    prep.add_argument("--case-count", type=int, default=20)
    prep.add_argument("--exclude-bundle", type=Path, action="append", default=[])
    prep.add_argument("--bundle-id", default="review-pilot-001")
    prep.add_argument("--min-frame-gap", type=int, default=750)
    gen = sub.add_parser("generate")
    for arg in ("input-dir", "output-dir", "checkpoint", "pin-json", "sam-repo"):
        gen.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.output_dir, args.anatomy, args.seed,
                image_count=args.image_count, case_count=args.case_count,
                exclude_bundles=args.exclude_bundle, bundle_id=args.bundle_id,
                min_frame_gap=args.min_frame_gap)
    else:
        generate(args.input_dir, args.output_dir, args.checkpoint, args.pin_json, args.sam_repo)


if __name__ == "__main__":
    main()
