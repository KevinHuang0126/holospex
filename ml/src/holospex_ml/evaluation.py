"""Compare checkpoints on unchanged original annotation grids.

Input resolution belongs to the checkpoint; the scoring grid belongs to the
dataset. Upsample logits before argmax, without HUD thresholds or polygons.
Case-equal scores first pool each video's pixels, then give each video equal
weight so a video with more annotated frames cannot dominate that average.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .dataset import encode_mask, read_image, read_semantic_mask
from .metrics import IGNORE_INDEX, confusion_matrix, metrics_from_confusion
from .model import load_checkpoint, resolve_device
from .training import (
    _check_evaluation_split, _check_ignore_policy, _manifest_digest,
    _samples, _validate_manifest, _video_ids,
)

SMALL_ANATOMY_IDS = (
    "cystic_duct", "cystic_artery", "cystic_plate",
    "hepatocystic_triangle_dissection",
)


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def _add_focus_scores(report: dict[str, Any]) -> dict[str, Any]:
    focused = [entry for entry in report["per_class"] if entry["structureId"] in SMALL_ANATOMY_IDS]
    report.update({
        "small_anatomy_macro_iou": _mean_defined([entry["iou"] for entry in focused]),
        "small_anatomy_macro_dice": _mean_defined([entry["dice"] for entry in focused]),
        "small_anatomy_classes_scored": sum(entry["iou"] is not None for entry in focused),
        "small_anatomy_class_ids": list(SMALL_ANATOMY_IDS),
    })
    return report


def _report_confusion(counts: np.ndarray, classes: list[dict], ignored_pixels: int = 0) -> dict:
    report = metrics_from_confusion(counts, classes, ignored_pixels=ignored_pixels)
    for index, entry in enumerate(report["per_class"]):
        true_positive = int(counts[index, index])
        predicted, support = entry["predicted_pixels"], entry["support_pixels"]
        entry.update({
            "true_positive_pixels": true_positive,
            "false_positive_pixels": predicted - true_positive,
            "false_negative_pixels": support - true_positive,
            "precision": true_positive / predicted if predicted else None,
            "recall": true_positive / support if support else None,
        })
    return _add_focus_scores(report)


def _case_equal(per_video: list[dict], classes: list[dict]) -> dict:
    """Average per-video class scores, including false-positive-only cases."""
    per_class = []
    for index, entry in enumerate(classes):
        cases = [video["per_class"][index] for video in per_video]
        per_class.append({
            **entry,
            **{metric: _mean_defined([case[metric] for case in cases])
               for metric in ("iou", "dice", "precision", "recall")},
            "present_truth_case_count": sum(case["support_pixels"] > 0 for case in cases),
            "false_positive_only_case_count": sum(case["support_pixels"] == 0 and case["predicted_pixels"] > 0 for case in cases),
            "absent_both_case_count": sum(case["iou"] is None for case in cases),
            "cases_scored": sum(case["iou"] is not None for case in cases),
            "precision_cases_scored": sum(case["precision"] is not None for case in cases),
            "recall_cases_scored": sum(case["recall"] is not None for case in cases),
        })
    return _add_focus_scores({
        "case_count": len(per_video), "per_class": per_class,
        "foreground_macro_iou": _mean_defined([entry["iou"] for entry in per_class[1:]]),
        "foreground_macro_dice": _mean_defined([entry["dice"] for entry in per_class[1:]]),
        "foreground_classes_scored": sum(entry["iou"] is not None for entry in per_class[1:]),
    })


def _logits_to_original_labels(logits: torch.Tensor, shape: tuple[int, int], num_classes: int) -> np.ndarray:
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != num_classes:
        raise ValueError("Model must return a single-frame 1×classes×height×width logit tensor.")
    if not torch.isfinite(logits).all():
        raise ValueError("Model returned nonfinite logits; refusing invalid scores.")
    # Resizing categorical argmax labels here would move boundaries differently
    # and make resolution comparisons depend on a different inference pipeline.
    original = F.interpolate(logits, size=shape, mode="bilinear", align_corners=False)
    return original.argmax(dim=1)[0].cpu().numpy()


def evaluate_original(
    checkpoint_path: Path, manifest: dict, split: str = "val",
    device: str = "auto", limit: int | None = None,
) -> dict:
    """Evaluate on original masks; test scoring requires explicitly split='test'.

    No loss is reported: this report compares pixel overlap and case behavior,
    independent of a checkpoint's training loss weighting.
    """
    classes = _validate_manifest(manifest)
    samples = _samples(manifest, split, limit)
    selected_device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path).resolve()
    with checkpoint_path.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    model, checkpoint = load_checkpoint(checkpoint_path, selected_device)
    if checkpoint["classes"] != classes:
        raise ValueError("Checkpoint and dataset class maps differ; refuse to score mislabeled outputs.")
    _check_evaluation_split(samples, split, checkpoint)
    ignored_sources = _check_ignore_policy(manifest, checkpoint)
    size = checkpoint["input_size"]
    normalization = checkpoint["normalization"]
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Checkpoint normalization requires three finite means and positive standard deviations.")
    mean_tensor = torch.from_numpy(mean).view(1, 3, 1, 1)
    std_tensor = torch.from_numpy(std).view(1, 3, 1, 1)
    counts = np.zeros((len(classes), len(classes)), dtype=np.int64)
    videos: dict[str, dict] = {}
    ignored_total, frame_records, original_sizes = 0, [], set()
    model.eval()
    with torch.inference_mode():
        for ordinal, sample in enumerate(samples, start=1):
            rgb = read_image(Path(sample["imagePath"]))
            mask = read_semantic_mask(Path(sample["maskPath"]))
            if rgb.shape[:2] != mask.shape:
                raise ValueError(f"Image/mask dimensions differ for {sample['imagePath']}.")
            target = encode_mask(mask, classes, ignore_source_ids=ignored_sources)
            height, width = target.shape
            original_sizes.add((width, height))
            image = Image.fromarray(rgb).resize((size["width"], size["height"]), Image.Resampling.BILINEAR)
            pixels = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
            pixels = ((pixels - mean_tensor) / std_tensor).to(selected_device)
            prediction = _logits_to_original_labels(model(pixels)["out"], target.shape, len(classes))
            frame_counts = confusion_matrix(target, prediction, len(classes))
            ignored = int(np.count_nonzero(target == IGNORE_INDEX))
            counts += frame_counts
            ignored_total += ignored
            video_id = str(sample["videoId"])
            video = videos.setdefault(video_id, {"counts": np.zeros_like(counts), "ignored": 0, "frames": 0})
            video["counts"] += frame_counts
            video["ignored"] += ignored
            video["frames"] += 1
            frame_records.append({
                "video_id": video_id, "frame_number": sample.get("frameNumber"),
                "timestamp_ms": sample.get("timestampMs"),
                "image_path": str(Path(sample["imagePath"]).resolve()),
                "mask_path": str(Path(sample["maskPath"]).resolve()),
                "width": width, "height": height,
                "scored_pixels": int(frame_counts.sum()), "ignored_pixels": ignored,
            })
            if ordinal == 1 or ordinal % 25 == 0 or ordinal == len(samples):
                print(f"evaluate-original split={split} frame={ordinal}/{len(samples)}", flush=True)
    if int(counts.sum()) == 0:
        raise ValueError("Selected evaluation frames contain only ignored pixels; no score is defined.")
    per_video = [{
        "video_id": video_id, "sample_count": video["frames"],
        **_report_confusion(video["counts"], classes, video["ignored"]),
    } for video_id, video in sorted(videos.items())]
    report = _report_confusion(counts, classes, ignored_total)
    equal = _case_equal(per_video, classes)
    for entry, equal_entry in zip(report["per_class"], equal["per_class"]):
        for field in ("present_truth_case_count", "false_positive_only_case_count", "absent_both_case_count", "cases_scored"):
            entry[field] = equal_entry[field]
    return {
        **report, "per_video": per_video, "case_equal": equal,
        "report_type": "original_resolution_segmentation_evaluation", "format_version": 1,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": split, "sample_count": len(samples), "case_count": len(videos),
        "model_id": checkpoint["model_id"], "model_version": checkpoint["model_version"],
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "architecture": checkpoint.get("architecture"), "input_size": size,
        "normalization": normalization, "device": str(selected_device),
        "limited_evaluation": limit is not None,
        "limited_training": checkpoint["training"]["limited_run"],
        "evaluation_video_ids": _video_ids(samples),
        "evaluation_manifest_sha256": _manifest_digest(manifest),
        "training_manifest_sha256": checkpoint["training"].get("manifest_sha256"),
        "manifest_digest_scope": "Canonical ordered JSON metadata; image and mask contents are not hashed",
        "ignore_index": IGNORE_INDEX, "ignore_source_ids": ignored_sources,
        "metric_resolution": {"basis": "original_mask_pixels", "sizes": [
            {"width": width, "height": height} for width, height in sorted(original_sizes)
        ]},
        "metric_policy": {
            "prediction": "bilinear resize logits to original mask shape before argmax; align_corners=False",
            "hud_filters_applied": False, "loss_included": False,
            "global": "Aggregate pixel confusion across selected frames; foreground macros exclude background",
            "case_equal": "Aggregate pixels within each video, then average each class score equally across defined video scores",
            "absent_classes": "Absent truth with false positives has IoU/Dice zero; absent truth and prediction has None and is excluded",
            "precision_recall": "Precision is None when no pixels are predicted; recall is None when truth is absent; case averages exclude undefined values",
            "small_anatomy": "Macro average over the four named anatomy classes with defined scores",
        },
        "frames": frame_records,
    }
