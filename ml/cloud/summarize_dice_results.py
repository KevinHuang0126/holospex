"""Audit and summarize all six collected runs from the frozen Dice iteration.

Read-only analysis: no downloads, training, test evaluation, or model selection.
All six completed attempts are required; incomplete/incomparable inputs fail.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


RUNS = [(f"dice-{2*i+j+1:03d}-{'ce' if j == 0 else 'gdl'}-s{seed}", seed,
         "ce" if j == 0 else "ce_generalized_dice") for i, seed in enumerate((42, 43, 44)) for j in (0, 1)]
FOREGROUND = ["gallbladder", "cystic_duct", "cystic_artery", "cystic_plate", "hepatocystic_triangle_dissection", "tool"]
SMALL = FOREGROUND[1:5]
FROZEN = {
    "architecture": "deeplabv3_mobilenet_v3_large", "model_id": "holospex-deeplabv3-mobilenetv3",
    "epochs": 40, "batch_size": 2, "learning_rate": 0.0003,
    "input_size": {"width": 672, "height": 384}, "device": "cuda",
    "pretrained": True, "initialization": "COCO_WITH_VOC_LABELS_V1",
    "class_weighting": "balanced", "batchnorm": "frozen",
    "augmentation": "none", "sampling": "uniform", "ignore_index": 255, "ignore_source_ids": [255],
    "train_samples": 343, "val_samples": 75, "limited_run": False,
    "selection_metric": "foreground_macro_iou", "metric_resolution": {"width": 672, "height": 384},
}
METRIC_PATHS = {
    "small_pooled_iou": ("small_anatomy_macro_iou",),
    "small_case_equal_iou": ("case_equal", "small_anatomy_macro_iou"),
    "foreground_iou": ("foreground_macro_iou",),
    "foreground_dice": ("foreground_macro_dice",),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def score(value, name):
    require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1,
            f"Invalid defined score: {name}")
    return value


def shared_config(config):
    value = deepcopy(config)
    for key in ("run_id", "seed", "loss", "dice_weight", "training_loss_reduction"):
        value.pop(key)
    for key in ("dice", "head_objective", "epoch_total"):
        value["loss_details"].pop(key)
    return value


def load_run(root, name, seed, objective, experiment, bundles):
    directory = Path(root) / name
    completion_path = directory / "cloud-completion.json"
    completion = read_json(completion_path)
    require(completion.get("state") == "completed" and completion.get("run_name") == name,
            f"{name}: require a completed matching attempt")
    require(completion.get("epochs_completed") == completion.get("epochs_requested") == 40, f"{name}: incomplete epoch budget")
    artifacts = completion["artifacts"]
    for logical_name, reference in artifacts.items():
        path = directory / logical_name
        require(not Path(logical_name).is_absolute() and ".." not in Path(logical_name).parts
                and path.resolve().is_relative_to(directory.resolve()) and not path.is_symlink(), f"{name}: unsafe artifact path")
        require(path.is_file() and path.stat().st_size == reference["bytes"] and digest(path) == reference["sha256"],
                f"{name}: artifact hash/size mismatch: {logical_name}")
    for required in ("train/config.json", "train/metrics-val-original.json", "train/history.json", "train/best.pt", "manifest.json", "environment.json"):
        require(required in artifacts, f"{name}: missing required artifact {required}")
    receipt = read_json(directory / "cloud-collection.json")
    require(receipt["completion_sha256"] == digest(completion_path) and receipt["metadata_paths_rewritten"] is False,
            f"{name}: collection receipt mismatch")
    config = read_json(directory / "train/config.json")
    metrics = read_json(directory / "train/metrics-val-original.json")
    manifest = read_json(directory / "manifest.json")
    history = read_json(directory / "train/history.json")
    for key, expected in {**FROZEN, "seed": seed, "loss": objective, "dice_weight": 0.0 if objective == "ce" else 1.0}.items():
        require(config.get(key) == expected, f"{name}: frozen setting {key} differs")
    require(config["loss_details"]["main_head_weight"] == 1.0 and config["loss_details"]["auxiliary_head_weight"] == 0.4,
            f"{name}: main/auxiliary objective differs")
    expected_reduction = "mean_over_target_class_weights" if objective == "ce" else "mean_over_optimizer_steps"
    require(config["training_loss_reduction"] == expected_reduction, f"{name}: epoch loss reporting differs")
    if objective == "ce_generalized_dice":
        policy = config["loss_details"]["dice"]
        require(policy["epsilon"] == 1e-6 and policy["variant"] == "foreground_present_classes_batch_pooled_generalized_soft_dice",
                f"{name}: Dice policy differs")
    else:
        require(config["loss_details"]["dice"] is None, f"{name}: CE control unexpectedly has Dice")
    require(config["classes"] == manifest["classes"] and [x["structureId"] for x in config["classes"]] == ["background"] + FOREGROUND,
            f"{name}: ontology mismatch")
    manifest_hash = canonical_digest(manifest)
    require(config["manifest_sha256"] == metrics["training_manifest_sha256"] == metrics["evaluation_manifest_sha256"] == manifest_hash,
            f"{name}: manifest identity mismatch")
    require(Counter(x["split"] for x in manifest["samples"]) == {"train": 343, "val": 75, "test": 74}, f"{name}: changed dataset counts")
    require(len(config["train_video_ids"]) == 30 and len(config["val_video_ids"]) == 10
            and not set(config["train_video_ids"]) & set(config["val_video_ids"]), f"{name}: invalid case split")
    for key, expected in {"split": "val", "sample_count": 75, "case_count": 10, "limited_evaluation": False,
                          "limited_training": False, "input_size": FROZEN["input_size"], "ignore_index": 255,
                          "ignore_source_ids": [255], "device": "cuda", "architecture": FROZEN["architecture"],
                          "metric_resolution": {"basis": "original_mask_pixels", "sizes": [{"width": 854, "height": 480}]}}.items():
        require(metrics.get(key) == expected, f"{name}: evaluation {key} differs")
    require(len(metrics["frames"]) == 75 and metrics["evaluation_video_ids"] == config["val_video_ids"], f"{name}: evaluation identities differ")
    require([x["epoch"] for x in history] == list(range(1, 41)), f"{name}: incomplete or reordered history")
    selected = max(history, key=lambda row: score(row["validation"]["foreground_macro_iou"], "selection IoU"))["epoch"]
    require(metrics["model_version"] == f"{config['run_id']}-epoch-{selected}", f"{name}: selected model version differs")
    require(metrics["checkpoint_sha256"] == artifacts["train/best.pt"]["sha256"], f"{name}: scored checkpoint differs")

    job_path = Path(experiment["config"])
    job = read_json(job_path)
    container = job["workerPoolSpecs"][0]["containerSpec"]
    environment = {x["name"]: x["value"] for x in container["env"]}
    for key, expected in {"HOLOSPEX_SOURCE_SHA256": bundles["source"]["sha256"], "HOLOSPEX_DATA_SHA256": bundles["data"]["sha256"],
                          "HOLOSPEX_LOSS": objective, "HOLOSPEX_SEED": str(seed), "HOLOSPEX_EPOCHS": "40",
                          "HOLOSPEX_RUN_NAME": name}.items():
        require(environment.get(key) == expected, f"{name}: submitted {key} differs")
    require(float(environment["HOLOSPEX_DICE_WEIGHT"]) == 1.0, f"{name}: submitted Dice coefficient differs")
    values = {}
    for label, keys in METRIC_PATHS.items():
        value = metrics
        for key in keys:
            value = value[key]
        values[label] = score(value, label)
    classes = {x["structureId"]: x for x in metrics["per_class"]}
    require(set(classes) == set(["background"] + FOREGROUND), f"{name}: metric classes differ")
    case_classes = {x["structureId"]: x for x in metrics["case_equal"]["per_class"]}
    require(math.isclose(values["small_case_equal_iou"], statistics.mean(score(case_classes[c]["iou"], f"{c} case-equal IoU") for c in SMALL), abs_tol=1e-12),
            f"{name}: case-equal aggregate disagrees with per-class scores")
    values["artery_recall"] = score(classes["cystic_artery"]["recall"], "artery recall")
    for label, ids, metric_key in (("small_pooled_iou", SMALL, "iou"), ("foreground_iou", FOREGROUND, "iou"), ("foreground_dice", FOREGROUND, "dice")):
        require(math.isclose(values[label], statistics.mean(score(classes[c][metric_key], f"{c} {metric_key}") for c in ids), abs_tol=1e-12),
                f"{name}: aggregate disagrees with per-class scores")
    result = {"run": name, "seed": seed, "loss": objective, "selected_epoch": selected, "metrics": values,
              "per_class": {c: {k: classes[c][k] for k in ("iou", "dice", "precision", "recall")} for c in FOREGROUND},
              "checkpoint_sha256": metrics["checkpoint_sha256"], "model_version": metrics["model_version"],
              "attempt": completion["attempt"], "job_name": experiment.get("job_name"),
              "completion_sha256": digest(completion_path), "config_sha256": digest(directory / "train/config.json"),
              "metrics_sha256": digest(directory / "train/metrics-val-original.json"), "job_config_sha256": digest(job_path),
              "all_artifact_hashes_verified": True}
    return result, {"shared_config": shared_config(config), "manifest": manifest, "frames": metrics["frames"],
                    "metric_policy": metrics["metric_policy"], "software_versions": config["software_versions"],
                    "runtime": read_json(directory / "environment.json"), "container": container["imageUri"],
                    "machine_spec": job["workerPoolSpecs"][0]["machineSpec"],
                    "dice_policy": config["loss_details"]["dice"]}


def distribution(values):
    require(all(value is not None and math.isfinite(value) for value in values), "Cannot average missing/nonfinite seed results")
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values), "n": len(values)}


def summarize(results_root, launch_manifest):
    launch = read_json(launch_manifest)
    experiments = {x["name"]: x for x in launch["experiments"]}
    require(len(launch["experiments"]) == 6 and set(experiments) == {x[0] for x in RUNS}, "Launch must contain exactly the frozen six runs")
    rows, common, dice_policy = [], None, None
    for name, seed, objective in RUNS:
        row, audit = load_run(results_root, name, seed, objective, experiments[name], launch["bundles"])
        policy = audit.pop("dice_policy")
        if policy is not None:
            require(dice_policy is None or policy == dice_policy, f"{name}: Dice policy changed across seeds")
            dice_policy = policy
        require(common is None or audit == common, f"{name}: settings, runtime, data, evaluation policy, or ordered frames differ")
        common = audit
        rows.append(row)
    pairs = [{"seed": rows[i]["seed"], "ce_run": rows[i]["run"], "dice_run": rows[i+1]["run"],
              "dice_minus_ce_percentage_points": {key: 100 * (rows[i+1]["metrics"][key] - rows[i]["metrics"][key]) for key in rows[i]["metrics"]}}
             for i in (0, 2, 4)]
    groups = {objective: {key: distribution([r["metrics"][key] for r in rows if r["loss"] == objective])
                          for key in rows[0]["metrics"]} for objective in ("ce", "ce_generalized_dice")}
    return {"report_type": "frozen_dice_six_run_validation_comparison", "generated_at": datetime.now(timezone.utc).isoformat(),
            "launch_manifest_sha256": digest(launch_manifest), "source_sha256": launch["bundles"]["source"]["sha256"],
            "data_sha256": launch["bundles"]["data"]["sha256"], "runs": rows, "paired_seeds": pairs,
            "group_summary_fraction_units": groups,
            "paired_delta_summary_percentage_points": {key: distribution([p["dice_minus_ce_percentage_points"][key] for p in pairs]) for key in rows[0]["metrics"]},
            "audit": {"all_six_complete": True, "all_shared_settings_match": True, "shared_config_sha256": canonical_digest(common["shared_config"]),
                      "manifest_sha256": canonical_digest(common["manifest"]), "software_versions": common["software_versions"],
                      "container": common["container"], "machine_spec": common["machine_spec"],
                      "dice_policy": dice_policy, "metric_policy": common["metric_policy"]},
            "interpretation": "Three paired seeds; validation reused, CUDA nondeterministic, no significance or model-promotion claim. All metrics are original-grid validation; test not evaluated by this script."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = summarize(args.results_root, args.launch_manifest)
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                output.write(encoded)
        else:
            print(encoded, end="")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Dice comparison failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
