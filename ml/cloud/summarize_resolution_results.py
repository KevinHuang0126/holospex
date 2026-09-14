"""Verify and compare the frozen 672/896/1120 validation-resolution runs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


SMALL = ["cystic_duct", "cystic_artery", "cystic_plate", "hepatocystic_triangle_dissection"]
FOREGROUND = ["gallbladder", *SMALL, "tool"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def valid_score(value, name):
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0 <= value <= 1, f"Invalid score: {name}")
    return value


def comparable_config(config):
    value = deepcopy(config)
    for key in ("run_id", "input_size", "metric_resolution", "class_weights",
                "class_weighting_details", "loss", "dice_weight", "loss_details"):
        value.pop(key, None)
    return value


def load_run(directory, name, width, height, *, require_completion=True):
    directory = Path(directory)
    completion = read(directory / "cloud-completion.json")
    require(completion.get("state") == "completed" and completion.get("run_name") == name,
            f"{name}: completed identity differs")
    require(completion.get("epochs_completed") == completion.get("epochs_requested") == 40,
            f"{name}: incomplete epoch budget")
    for logical, record in completion["artifacts"].items():
        path = directory / logical
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()),
                f"{name}: missing or unsafe artifact {logical}")
        require(path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
                f"{name}: artifact identity differs {logical}")
    config = read(directory / "train/config.json")
    metrics = read(directory / "train/metrics-val-original.json")
    history = read(directory / "train/history.json")
    manifest = read(directory / "manifest.json")
    expected_size = {"width": width, "height": height}
    for key, expected in {
        "epochs": 40, "batch_size": 2, "learning_rate": 0.0003,
        "input_size": expected_size, "architecture": "deeplabv3_mobilenet_v3_large",
        "augmentation": "none", "sampling": "uniform", "seed": 42,
        "class_weighting": "balanced", "train_samples": 343, "val_samples": 75,
    }.items():
        require(config.get(key) == expected, f"{name}: configuration differs for {key}")
    require(config.get("loss", "ce") == "ce" and config.get("dice_weight", 0.0) in (0, 0.0, None),
            f"{name}: not a balanced-CE run")
    require(metrics.get("input_size") == expected_size and metrics.get("split") == "val"
            and metrics.get("sample_count") == 75 and metrics.get("case_count") == 10,
            f"{name}: evaluation identity differs")
    require(metrics.get("metric_resolution") == {"basis": "original_mask_pixels", "sizes": [{"width": 854, "height": 480}]},
            f"{name}: not original-grid validation")
    require([row["epoch"] for row in history] == list(range(1, 41)), f"{name}: incomplete history")
    selected = max(history, key=lambda row: valid_score(row["validation"]["foreground_macro_iou"], "selection"))["epoch"]
    require(metrics["model_version"] == f"{config['run_id']}-epoch-{selected}", f"{name}: wrong selected checkpoint")
    require(metrics["checkpoint_sha256"] == completion["artifacts"]["train/best.pt"]["sha256"],
            f"{name}: evaluated checkpoint differs")
    classes = {row["structureId"]: row for row in metrics["per_class"]}
    case_classes = {row["structureId"]: row for row in metrics["case_equal"]["per_class"]}
    require(set(classes) == {"background", *FOREGROUND}, f"{name}: class map differs")
    values = {
        "small_pooled_iou": valid_score(metrics["small_anatomy_macro_iou"], "small pooled IoU"),
        "small_case_equal_iou": valid_score(metrics["case_equal"]["small_anatomy_macro_iou"], "small case-equal IoU"),
        "foreground_iou": valid_score(metrics["foreground_macro_iou"], "foreground IoU"),
        "foreground_dice": valid_score(metrics["foreground_macro_dice"], "foreground Dice"),
        "artery_recall": valid_score(classes["cystic_artery"]["recall"], "artery recall"),
    }
    require(math.isclose(values["small_pooled_iou"], sum(classes[key]["iou"] for key in SMALL) / 4, abs_tol=1e-12),
            f"{name}: pooled small aggregate differs")
    require(math.isclose(values["small_case_equal_iou"], sum(case_classes[key]["iou"] for key in SMALL) / 4, abs_tol=1e-12),
            f"{name}: case-equal small aggregate differs")
    return {
        "name": name, "input_size": expected_size, "selected_epoch": selected,
        "metrics": values,
        "per_class": {key: {metric: classes[key][metric] for metric in ("iou", "dice", "precision", "recall")}
                      for key in FOREGROUND},
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "epoch_seconds_total": sum(row["duration_seconds"] for row in history),
        "epoch_seconds_mean": sum(row["duration_seconds"] for row in history) / 40,
        "derived_class_weights": config["class_weights"],
        "artifact_hashes_verified": True,
    }, {
        "config": comparable_config(config), "manifest": manifest,
        "frames": metrics["frames"], "metric_policy": metrics["metric_policy"],
        "evaluation_video_ids": metrics["evaluation_video_ids"],
    }


def summarize(results_root, launch_manifest):
    root = Path(results_root)
    launch = read(launch_manifest)
    expected = [("cloud-002-longer", 672, 384),
                ("resolution-001-896x512", 896, 512),
                ("resolution-002-1120x640", 1120, 640)]
    rows, audit = [], None
    for name, width, height in expected:
        row, current = load_run(root / name, name, width, height)
        require(audit is None or current == audit,
                f"{name}: data, evaluation frames/policy, or non-resolution settings differ")
        audit = current
        rows.append(row)
    baseline = rows[0]
    for row in rows:
        row["delta_vs_672_percentage_points"] = {
            key: 100 * (value - baseline["metrics"][key]) for key, value in row["metrics"].items()
        }
    return {
        "report_type": "frozen_resolution_validation_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "launch_manifest_sha256": digest(launch_manifest),
        "runs": rows,
        "audit": {"all_runs_complete": True, "all_artifacts_verified": True,
                  "shared_manifest_sha256": canonical(audit["manifest"]),
                  "resolution_and_derived_class_weights_are_only_training_differences": True,
                  "test_split_evaluated": False},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = summarize(args.results_root, args.launch_manifest)
        with args.output.open("x") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Resolution comparison failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
