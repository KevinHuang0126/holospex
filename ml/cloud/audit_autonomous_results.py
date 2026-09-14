"""Read-only verification of a collected autonomous run before reporting success.

The success metric is six-class foreground macro IoU on the fixed 75 native
Endoscapes validation masks. No model is loaded and no dataset is downloaded.
Artifact hashes bind the report to the collected checkpoint, not to a new
independent inference run. Invalid or incomplete evidence raises ValueError.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from holospex_ml.metrics import metrics_from_confusion


ONTOLOGY = [{"index": i, "sourceId": source, "structureId": structure}
            for i, (source, structure) in enumerate(zip(
                (0, 5, 4, 3, 1, 2, 6),
                ("background", "gallbladder", "cystic_duct", "cystic_artery",
                 "cystic_plate", "hepatocystic_triangle_dissection", "tool")))]
# Frozen from the original usable Seg50 validation split, before this search.
VALIDATION_FRAMES = {
    "126": list(range(11550, 13801, 750)), "131": list(range(40375, 53876, 750)),
    "137": [7125, 7875], "141": list(range(39075, 41326, 750)),
    "142": list(range(28750, 37001, 750)), "144": [16425, 17175],
    "146": [12525, 13275],
    "153": [x for x in range(18450, 33451, 750) if x != 32700],
    "156": [43575, 44325], "159": list(range(55550, 60801, 750)),
}
VALIDATION_CASES = sorted(VALIDATION_FRAMES)
VALIDATION_IDENTITIES = {(case, frame) for case, frames in VALIDATION_FRAMES.items() for frame in frames}
VALIDATION_TRUTH_SUPPORT = [21997774, 5583003, 608598, 114832, 131815, 42710, 2204677]
NATIVE_SIZE = {"width": 854, "height": 480}
NATIVE_IGNORE_PIXELS = 60591
TARGET_IOU = 0.75


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same(actual, expected, context):
    if expected is None:
        _require(actual is None, f"{context}: expected an undefined score")
    elif isinstance(expected, float):
        _require(type(actual) in (int, float) and math.isfinite(actual)
                 and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), f"{context}: score mismatch")
    else:
        _require(actual == expected, f"{context}: mismatch")


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _verify_metrics(report, context):
    counts = report.get("confusion_matrix")
    _require(isinstance(counts, list) and len(counts) == 7
             and all(isinstance(row, list) and len(row) == 7 for row in counts)
             and all(type(value) is int and 0 <= value <= 30744000 for row in counts for value in row),
             f"{context}: invalid confusion counts")
    expected = metrics_from_confusion(np.array(counts, dtype=np.int64), ONTOLOGY,
                                      ignored_pixels=report.get("ignored_pixels"))
    _require(len(report.get("per_class", [])) == 7, f"{context}: incomplete ontology")
    for index, row in enumerate(expected["per_class"]):
        actual = report["per_class"][index]
        tp = counts[index][index]
        row.update(true_positive_pixels=tp, false_positive_pixels=row["predicted_pixels"] - tp,
                   false_negative_pixels=row["support_pixels"] - tp,
                   precision=tp / row["predicted_pixels"] if row["predicted_pixels"] else None,
                   recall=tp / row["support_pixels"] if row["support_pixels"] else None)
        for key, value in row.items():
            _same(actual.get(key), value, f"{context} class {index} {key}")
    for key in ("foreground_macro_iou", "foreground_macro_dice", "foreground_classes_scored",
                "pixel_accuracy", "scored_pixels", "ignored_pixels", "total_pixels"):
        _same(report.get(key), expected[key], f"{context} {key}")
    for metric in ("iou", "dice"):
        key = f"small_anatomy_macro_{metric}"
        expected[key] = _mean([row[metric] for row in expected["per_class"][2:6]])
        _same(report.get(key), expected[key], f"{context} {key}")
    _same(report.get("small_anatomy_class_ids"), [row["structureId"] for row in ONTOLOGY[2:6]], context)
    _same(report.get("small_anatomy_classes_scored"), sum(row["iou"] is not None for row in expected["per_class"][2:6]), context)
    return expected


def _summarize_run(run_root) -> dict:
    root = Path(run_root).resolve()
    completion_path = root / "cloud-completion.json"
    completion = _read_json(completion_path)
    receipt = _read_json(root / "cloud-collection.json")
    _require(completion.get("state") == "completed", "Run is not completed")
    _require(receipt.get("metadata_paths_rewritten") is False, "Collected metadata was rewritten")
    _same(receipt.get("completion_sha256"), _sha256(completion_path), "Completion receipt SHA-256")
    _same(receipt.get("completion_bytes"), completion_path.stat().st_size, "Completion receipt bytes")
    artifacts = completion.get("artifacts", {})
    _same(receipt.get("artifact_count"), len(artifacts), "Collection artifact count")
    required = ("train/best.pt", "train/config.json", "train/history.json", "train/metrics-val-original.json", "manifest.json")
    hashes = {}
    for name in required:
        reference, path = artifacts.get(name), root / name
        _require(isinstance(reference, dict) and path.is_file() and not path.is_symlink(), f"Missing regular artifact: {name}")
        _require(root in path.resolve().parents, f"Artifact escapes run root: {name}")
        hashes[name] = _sha256(path)
        _same(reference.get("sha256"), hashes[name], f"Artifact SHA-256 {name}")
        _same(reference.get("bytes"), path.stat().st_size, f"Artifact bytes {name}")
    config = _read_json(root / "train/config.json")
    history = _read_json(root / "train/history.json")
    metrics = _read_json(root / "train/metrics-val-original.json")
    manifest = _read_json(root / "manifest.json")
    _same(config.get("classes"), ONTOLOGY, "Training ontology")
    _same(manifest.get("classes"), ONTOLOGY, "Manifest ontology")
    _require(config.get("limited_run") is False and config.get("limit_train") is None
             and config.get("limit_val") is None, "Limited training is not eligible")
    _require(metrics.get("limited_evaluation") is False and metrics.get("limited_training") is False,
             "Limited evaluation is not eligible")
    for data, label, index_key, source_key in ((metrics, "Metrics", "ignore_index", "ignore_source_ids"),
                                              (config, "Config", "ignore_index", "ignore_source_ids"),
                                              (manifest, "Manifest", "ignoreIndex", "ignoreSourceIds")):
        _same(data.get(index_key), 255, f"{label} ignore index")
        _same(data.get(source_key), [255], f"{label} ignore policy")
    _require(metrics.get("report_type") == "original_resolution_segmentation_evaluation"
             and metrics.get("split") == "val", "Require original-grid validation report")
    _same(metrics.get("metric_resolution"), {"basis": "original_mask_pixels", "sizes": [NATIVE_SIZE]}, "Native metric grid")
    policy = metrics.get("metric_policy", {})
    _require(policy.get("hud_filters_applied") is False and policy.get("prediction") ==
             "bilinear resize logits to original mask shape before argmax; align_corners=False", "Metric prediction policy differs")
    _same(metrics.get("checkpoint_sha256"), hashes["train/best.pt"], "Metric checkpoint SHA-256")
    _same(metrics.get("architecture"), config.get("architecture"), "Model architecture")
    _same(metrics.get("model_id"), config.get("model_id"), "Model identity")
    _same(metrics.get("input_size"), config.get("input_size"), "Model input size")
    canonical_manifest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    _same(config.get("manifest_sha256"), canonical_manifest, "Training manifest digest")
    _same(metrics.get("training_manifest_sha256"), canonical_manifest, "Evaluation training manifest digest")
    _same(metrics.get("evaluation_manifest_sha256"), canonical_manifest, "Evaluation manifest digest")
    samples = manifest.get("samples", [])
    identities = [(str(row["videoId"]), row["frameNumber"]) for row in samples]
    _require(len(identities) == len(set(identities)), "Duplicate manifest frame identities")
    _require(all(row.get("split") in {"train", "val", "test"} for row in samples), "Unknown manifest split")
    cases = {split: sorted({str(row["videoId"]) for row in samples if row["split"] == split}) for split in ("train", "val", "test")}
    _require(cases["train"] and not (set(cases["train"]) & set(cases["val"]) or set(cases["train"]) & set(cases["test"])
                                  or set(cases["val"]) & set(cases["test"])), "Train/validation/test case leakage")
    _same(cases["val"], VALIDATION_CASES, "Fixed validation cases")
    _same(config.get("train_video_ids"), cases["train"], "Training cases")
    _same(config.get("val_video_ids"), VALIDATION_CASES, "Selection cases")
    _same(metrics.get("evaluation_video_ids"), VALIDATION_CASES, "Evaluation cases")
    _same(config.get("train_samples"), sum(row["split"] == "train" for row in samples), "Training sample count")
    _same(config.get("val_samples"), 75, "Training validation count")
    _same({(str(row["videoId"]), row["frameNumber"]) for row in samples if row["split"] == "val"}, VALIDATION_IDENTITIES, "Fixed validation frame identities")
    frames = metrics.get("frames", [])
    _require(len(frames) == metrics.get("sample_count") == 75 and metrics.get("case_count") == 10, "Require all 75 validation frames and ten cases")
    _same({(row["video_id"], row["frame_number"]) for row in frames}, VALIDATION_IDENTITIES, "Scored frame identities")
    for frame in frames:
        _require({key: frame.get(key) for key in NATIVE_SIZE} == NATIVE_SIZE, "Frame is not native resolution")
        _require(type(frame.get("scored_pixels")) is int and type(frame.get("ignored_pixels")) is int
                 and frame["scored_pixels"] >= 0 and frame["ignored_pixels"] >= 0
                 and frame["scored_pixels"] + frame["ignored_pixels"] == 854 * 480, "Frame pixel accounting mismatch")
    pooled = _verify_metrics(metrics, "Pooled")
    _same([row["support_pixels"] for row in pooled["per_class"]], VALIDATION_TRUTH_SUPPORT, "Fixed native validation truth counts")
    _same(pooled["ignored_pixels"], NATIVE_IGNORE_PIXELS, "Fixed native ignored pixels")
    videos = metrics.get("per_video", [])
    _same([row.get("video_id") for row in videos], VALIDATION_CASES, "Per-video case identities")
    recomputed = []
    for video in videos:
        selected = [frame for frame in frames if frame["video_id"] == video["video_id"]]
        _same(video.get("sample_count"), len(selected), "Per-video sample count")
        value = _verify_metrics(video, f"Case {video['video_id']}")
        for key in ("scored_pixels", "ignored_pixels"):
            _same(value[key], sum(frame[key] for frame in selected), f"Per-video {key}")
        recomputed.append(value)
    _same(np.sum([row["confusion_matrix"] for row in recomputed], axis=0).tolist(), pooled["confusion_matrix"], "Per-video confusion sum")
    equal = metrics.get("case_equal", {})
    _same(equal.get("case_count"), 10, "Case-equal count")
    _require(len(equal.get("per_class", [])) == 7, "Case-equal ontology incomplete")
    equal_classes = []
    for index, actual in enumerate(equal["per_class"]):
        expected = {**ONTOLOGY[index], **{key: _mean([video["per_class"][index][key] for video in recomputed])
                                        for key in ("iou", "dice", "precision", "recall")}}
        for key, value in expected.items():
            _same(actual.get(key), value, f"Case-equal class {index} {key}")
        equal_classes.append(expected)
    for prefix, subset in (("foreground", equal_classes[1:]), ("small_anatomy", equal_classes[2:6])):
        for metric in ("iou", "dice"):
            _same(equal.get(f"{prefix}_macro_{metric}"), _mean([row[metric] for row in subset]), f"Case-equal {prefix} {metric}")
    epochs, completed_epochs = config.get("epochs"), completion.get("epochs_completed")
    _require(type(epochs) is int and epochs > 0 and type(completed_epochs) is int
             and 1 <= completed_epochs <= epochs and isinstance(history, list)
             and len(history) == completed_epochs, "Incomplete recorded training history")
    _same(completion.get("epochs_requested"), epochs, "Requested epoch count")
    duration_limited = completed_epochs < epochs
    if duration_limited:
        duration_limit, actual_duration = config.get("max_duration_seconds"), completion.get("training_duration_seconds")
        _require(completion.get("budget_exhausted") is True and completion.get("stop_reason") == "max_duration_seconds",
                 "Incomplete budget lacks a verified duration stop")
        _require(type(duration_limit) in (int, float) and math.isfinite(duration_limit) and duration_limit > 0
                 and type(actual_duration) in (int, float) and math.isfinite(actual_duration)
                 and actual_duration >= duration_limit, "Duration stop did not exhaust its configured time budget")
    else:
        _require(completion.get("budget_exhausted", False) is False, "Full epoch budget incorrectly marked exhausted")
    _same([row.get("epoch") for row in history], list(range(1, completed_epochs + 1)), "Sequential completed epochs")
    scores = [row.get("validation", {}).get("foreground_macro_iou") for row in history]
    _require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in scores), "Invalid selection scores")
    _same(config.get("selection_metric"), "foreground_macro_iou", "Selection metric")
    selected_epoch = scores.index(max(scores)) + 1
    _same(metrics.get("model_version"), f"{config['run_id']}-epoch-{selected_epoch}", "Selected checkpoint epoch")
    primary = pooled["foreground_macro_iou"]
    _require(pooled["foreground_classes_scored"] == 6 and primary is not None, "Require all six foreground classes")
    return {"run_name": completion.get("run_name", root.name), "run_root": str(root),
            "foreground_macro_iou": primary, "small_pooled_iou": pooled["small_anatomy_macro_iou"],
            "small_case_equal_iou": _mean([row["iou"] for row in equal_classes[2:6]]),
            "selected_epoch": selected_epoch, "epochs_completed": completed_epochs, "epochs_requested": epochs,
            "duration_limited": duration_limited,
            "checkpoint": str(root / "train/best.pt"), "checkpoint_sha256": hashes["train/best.pt"],
            "sample_count": 75, "case_count": 10, "metric_resolution": metrics["metric_resolution"],
            "train_video_ids": cases["train"], "validation_video_ids": VALIDATION_CASES,
            "per_class": pooled["per_class"], "verified": True, "target_iou": TARGET_IOU,
            "target_reached": primary >= TARGET_IOU,
            "audit": {"artifact_hashes_verified": len(required), "native_grid_verified": True,
                      "fixed_validation_identities_verified": True, "confusion_metrics_recomputed": True,
                      "split_cases_disjoint": True, "model_loaded": False}}


def summarize_run(run_root) -> dict:
    """Return verified fractional scores; raise ValueError for invalid evidence."""
    try:
        return _summarize_run(run_root)
    except (OSError, KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError(f"Incomplete or invalid collected run evidence: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(summarize_run(args.run_root), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
