"""Audit the six frozen reviewed-data runs and compare their validation scores.

Reads collected artifacts only. Base runs use 42 epochs and added-data runs 40
to approximately match optimizer steps, fixed before seeing any results. Every
seed and both arms must finish before a summary is emitted. This script never
loads a model, evaluates test data, or promotes a checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics


RUNS = [(f"reviewed-{2 * i + j + 1:03d}-{arm}-s{seed}", seed, arm)
        for i, seed in enumerate((42, 43, 44)) for j, arm in enumerate(("base", "added"))]
FOREGROUND = ["gallbladder", "cystic_duct", "cystic_artery", "cystic_plate", "hepatocystic_triangle_dissection", "tool"]
SMALL = FOREGROUND[1:5]
ONTOLOGY = [{"index": i, "sourceId": source, "structureId": structure}
            for i, (source, structure) in enumerate(zip((0, 5, 4, 3, 1, 2, 6), ["background"] + FOREGROUND))]
ARM = {"base": {"epochs": 42, "train_samples": 343, "train_cases": 30},
       "added": {"epochs": 40, "train_samples": 361, "train_cases": 48}}
FROZEN = {
    "architecture": "deeplabv3_mobilenet_v3_large", "model_id": "holospex-deeplabv3-mobilenetv3",
    "batch_size": 2, "learning_rate": 0.0003, "input_size": {"width": 672, "height": 384}, "device": "cuda",
    "pretrained": True, "initialization": "COCO_WITH_VOC_LABELS_V1", "class_weighting": "balanced",
    "batchnorm": "frozen", "augmentation": "none", "sampling": "uniform", "ignore_index": 255,
    "ignore_source_ids": [255], "val_samples": 75, "limited_run": False, "limit_train": None, "limit_val": None,
    "selection_metric": "foreground_macro_iou", "metric_resolution": {"width": 672, "height": 384},
    "loss": "ce", "dice_weight": 0.0, "training_loss_reduction": "mean_over_target_class_weights",
    "validation_loss_weighting": "none",
}
METRIC_PATHS = {"small_pooled_iou": ("small_anatomy_macro_iou",),
                "small_case_equal_iou": ("case_equal", "small_anatomy_macro_iou"),
                "foreground_iou": ("foreground_macro_iou",), "foreground_dice": ("foreground_macro_dice",)}
CLASS_METRICS = ("iou", "dice", "precision", "recall")
PATH_FIELDS = {"imagePath", "maskPath", "reviewProvenancePath", "image_path", "mask_path"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, f"Duplicate JSON key: {key}")
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def record_digest(value):
    return hashlib.sha256((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()


def score(value, name, nullable=False):
    if nullable and value is None:
        return None
    require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1,
            f"Invalid defined score: {name}")
    return value


def normalize_paths(value):
    """Ignore machine-specific prefixes, preserving the storage folder/filename.

    Collected manifests retain their original cloud paths. Base sample metadata
    can be compared without rebasing or editing those evidence files. Pixel
    identity is established separately by the submitted hashed data bundles.
    """
    if isinstance(value, list):
        return [normalize_paths(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in PATH_FIELDS:
            require(isinstance(item, str) and ".." not in PurePosixPath(item).parts, f"Unsafe dataset path: {item}")
            parts = PurePosixPath(item).parts
            require(len(parts) >= 2, f"Dataset path lacks folder identity: {item}")
            result[key] = "/".join(parts[-2:])
        else:
            result[key] = normalize_paths(item)
    return result


def prepared_manifest_metadata(manifest):
    """Remove exactly the cloud-staging path changes, retaining other metadata."""
    value = normalize_paths(deepcopy(manifest))
    value.pop("root", None)
    if "reviewedPartialProvenance" in value:
        value["reviewedPartialProvenance"].pop("preparationPath", None)
    return value


def verify_prepared_content(directory, manifest, input_audit, verification, name):
    """Bind bytes actually seen by training to the frozen package file inventory.

    A consistent mask change in every added run would pass comparisons between
    seeds. Joining through exact source paths also binds those new masks to the
    reviewed package, whose snapshot hash is fixed in the launch manifest.
    """
    package = read_json(directory / "prepared-package.json")
    source = read_json(directory / "manifest-source.json")
    require(package["manifestSha256"] == verification["sourceManifestSha256"]
            and package["baseManifestSha256"] == verification["baseManifestSha256"],
            f"{name}: package source manifest hashes disagree")
    require(prepared_manifest_metadata(source) == prepared_manifest_metadata(manifest),
            f"{name}: runtime manifest metadata differs from frozen prepared source")
    records = package["files"]
    files = {row["sourcePath"]: row for row in records}
    require(len(files) == len(records), f"{name}: duplicate prepared package source path")
    require(verification["verifiedFileCount"] == len(records), f"{name}: verified package file count differs")
    offsets = {split: 0 for split in ("train", "val", "test")}
    for sample in source["samples"]:
        split = sample["split"]
        actual = input_audit["samples"][split][offsets[split]]
        offsets[split] += 1
        for prefix in ("image", "mask"):
            source_path = sample[f"{prefix}Path"]
            require(source_path in files, f"{name}: sample absent from frozen prepared file inventory")
            require(actual[f"{prefix}Sha256"] == files[source_path]["sha256"],
                    f"{name}: actual training {prefix} hash differs from frozen prepared package: {source_path}")


def shared_config(config):
    """Only permit the planned epoch budget and data-derived statistics to vary."""
    value = deepcopy(config)
    for key in ("run_id", "seed", "epochs", "train_samples", "train_video_ids", "manifest_sha256",
                "class_weights", "dataset"):
        value.pop(key)
    details = value["class_weighting_details"]
    for key in ("sample_count", "counts", "scored_pixels", "ignored_pixels", "frequencies",
                "raw_inverse_sqrt_weights", "foreground_mean_raw_weight", "weights"):
        details.pop(key)
    for key in ("draws_per_epoch", "video_frame_counts"):
        value["sampling_details"].pop(key)
    return value


def validate_manifest(manifest, config, arm, name):
    require(config["classes"] == manifest["classes"] == ONTOLOGY, f"{name}: ontology mismatch")
    require(manifest.get("ignoreIndex") == 255 and manifest.get("ignoreSourceIds") == [255], f"{name}: manifest ignore policy differs")
    samples = normalize_paths(manifest["samples"])
    expected = {"train": ARM[arm]["train_samples"], "val": 75, "test": 74}
    require(Counter(x["split"] for x in samples) == expected, f"{name}: changed dataset counts")
    identities = [(str(x["videoId"]), x["frameNumber"]) for x in samples]
    require(len(set(identities)) == len(identities), f"{name}: duplicate frame identity")
    cases = {split: sorted({str(x["videoId"]) for x in samples if x["split"] == split}) for split in expected}
    require({split: len(ids) for split, ids in cases.items()} == {"train": ARM[arm]["train_cases"], "val": 10, "test": 10},
            f"{name}: wrong case counts")
    require(not (set(cases["train"]) & set(cases["val"]) or set(cases["train"]) & set(cases["test"])
                 or set(cases["val"]) & set(cases["test"])), f"{name}: case leakage across splits")
    require(config["train_video_ids"] == cases["train"] and config["val_video_ids"] == cases["val"], f"{name}: config case identities differ")
    counts = dict(sorted(Counter(str(x["videoId"]) for x in samples if x["split"] == "train").items()))
    sampling = config["sampling_details"]
    require(sampling["draws_per_epoch"] == expected["train"] and sampling["video_frame_counts"] == counts
            and sampling["replacement"] is False and sampling["weight_formula"] == "uniform_shuffle_without_replacement",
            f"{name}: sampling differs from complete uniform training")
    weighting = config["class_weighting_details"]
    require(weighting["sample_count"] == expected["train"] and weighting["weights"] == config["class_weights"]
            and len(weighting["counts"]) == len(ONTOLOGY) and sum(weighting["counts"]) == weighting["scored_pixels"],
            f"{name}: class weighting statistics disagree")
    require(weighting["scored_pixels"] + weighting["ignored_pixels"] == expected["train"] * 672 * 384,
            f"{name}: training pixel accounting differs")
    return samples, cases


def validate_commands(commands, name):
    evaluations = []
    for _, command, _ in commands:
        if "evaluate" in command or "evaluate-original" in command:
            require("evaluate-original" in command and "--split" in command
                    and command[command.index("--split") + 1] == "val", f"{name}: test or unexpected evaluation command")
            evaluations.append(command)
    require(len(evaluations) == 1, f"{name}: require exactly one recorded original-grid validation evaluation")


def confusion_scores(matrix):
    """Independent integer-count derivation; does not import the evaluator."""
    require(isinstance(matrix, list) and len(matrix) == 7 and all(isinstance(row, list) and len(row) == 7 for row in matrix),
            "Confusion matrix must be 7 by 7")
    require(all(type(value) is int and value >= 0 for row in matrix for value in row), "Confusion counts must be nonnegative integers")
    result = []
    for index, entry in enumerate(ONTOLOGY):
        tp, support, predicted = matrix[index][index], sum(matrix[index]), sum(row[index] for row in matrix)
        union = support + predicted - tp
        result.append({**entry, "support_pixels": support, "predicted_pixels": predicted, "true_positive_pixels": tp,
                       "false_positive_pixels": predicted - tp, "false_negative_pixels": support - tp,
                       "iou": tp / union if union else None, "dice": 2 * tp / (support + predicted) if support + predicted else None,
                       "precision": tp / predicted if predicted else None, "recall": tp / support if support else None})
    return result


def mean_defined(values):
    defined = [x for x in values if x is not None]
    return statistics.mean(defined) if defined else None


def assert_numeric(actual, expected, context):
    if expected is None:
        require(actual is None, f"{context}: undefined score disagrees with confusion")
    else:
        require(not isinstance(actual, bool) and isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), f"{context}: value disagrees with confusion")


def verify_confusion(metrics, name):
    frames = metrics["frames"]
    require(len(metrics["per_video"]) == 10, f"{name}: expected ten per-video confusion matrices")
    require([x["video_id"] for x in metrics["per_video"]] == metrics["evaluation_video_ids"], f"{name}: per-video confusion identities differ")
    global_sum = [[0] * 7 for _ in range(7)]
    video_classes = []
    for report in [*metrics["per_video"], metrics]:
        matrix = report["confusion_matrix"]
        expected_classes = confusion_scores(matrix)
        require(len(report["per_class"]) == 7, f"{name}: wrong confusion class count")
        for actual, expected in zip(report["per_class"], expected_classes):
            require(all(actual[key] == expected[key] for key in ("index", "sourceId", "structureId")), f"{name}: confusion ontology differs")
            for key in (*CLASS_METRICS, "support_pixels", "predicted_pixels", "true_positive_pixels", "false_positive_pixels", "false_negative_pixels"):
                assert_numeric(actual[key], expected[key], f"{name} {expected['structureId']} {key}")
        for key, selected, field in (("foreground_macro_iou", expected_classes[1:], "iou"),
                                      ("foreground_macro_dice", expected_classes[1:], "dice"),
                                      ("small_anatomy_macro_iou", expected_classes[2:6], "iou"),
                                      ("small_anatomy_macro_dice", expected_classes[2:6], "dice")):
            assert_numeric(report[key], mean_defined([x[field] for x in selected]), f"{name} {key}")
        selected_frames = [x for x in frames if x["video_id"] == report["video_id"]] if "video_id" in report else frames
        scored = sum(sum(row) for row in matrix)
        require(report["sample_count"] == len(selected_frames), f"{name}: per-video sample count differs")
        require(report["scored_pixels"] == scored == sum(x["scored_pixels"] for x in selected_frames), f"{name}: scored pixel accounting differs")
        require(report["ignored_pixels"] == sum(x["ignored_pixels"] for x in selected_frames), f"{name}: ignored pixel accounting differs")
        require(report["total_pixels"] == scored + report["ignored_pixels"] == sum(x["width"] * x["height"] for x in selected_frames),
                f"{name}: total pixel accounting differs")
        if "video_id" in report:
            video_classes.append(expected_classes)
            for i in range(7):
                for j in range(7):
                    global_sum[i][j] += matrix[i][j]
    require(global_sum == metrics["confusion_matrix"], f"{name}: per-video confusion sum differs from global")
    equal = metrics["case_equal"]
    require(equal["case_count"] == len(video_classes) and len(equal["per_class"]) == 7, f"{name}: case-equal count differs")
    expected_equal = []
    for index, actual in enumerate(equal["per_class"]):
        require(all(actual[key] == ONTOLOGY[index][key] for key in ("index", "sourceId", "structureId")), f"{name}: case-equal ontology differs")
        expected = {key: mean_defined([x[index][key] for x in video_classes]) for key in CLASS_METRICS}
        for key, value in expected.items():
            assert_numeric(actual[key], value, f"{name} case-equal {ONTOLOGY[index]['structureId']} {key}")
        expected_equal.append(expected)
    for key, selected, field in (("foreground_macro_iou", expected_equal[1:], "iou"), ("foreground_macro_dice", expected_equal[1:], "dice"),
                                  ("small_anatomy_macro_iou", expected_equal[2:6], "iou"), ("small_anatomy_macro_dice", expected_equal[2:6], "dice")):
        assert_numeric(equal[key], mean_defined([x[field] for x in selected]), f"{name} case-equal {key}")


def load_run(root, name, seed, arm, experiment, bundles):
    directory = Path(root) / name
    completion_path = directory / "cloud-completion.json"
    completion = read_json(completion_path)
    epochs = ARM[arm]["epochs"]
    require(completion.get("state") == "completed" and completion.get("run_name") == name, f"{name}: require a completed matching attempt")
    require(completion.get("epochs_completed") == completion.get("epochs_requested") == epochs, f"{name}: incomplete or changed epoch budget")
    require(completion["attempt_uri"].startswith(experiment["output_uri"].rstrip("/") + "/attempts/"), f"{name}: output attempt differs from launch")
    artifacts = completion["artifacts"]
    for logical, reference in artifacts.items():
        path = directory / logical
        require(not Path(logical).is_absolute() and ".." not in Path(logical).parts
                and path.resolve().is_relative_to(directory.resolve()) and not path.is_symlink(), f"{name}: unsafe artifact path")
        require(path.is_file() and path.stat().st_size == reference["bytes"] and digest(path) == reference["sha256"],
                f"{name}: artifact hash/size mismatch: {logical}")
        require(not ("metric" in logical.lower() and "test" in logical.lower()), f"{name}: unexpected test metrics artifact")
    for logical in ("train/config.json", "train/metrics-val-original.json", "train/history.json", "train/best.pt",
                    "manifest.json", "environment.json", "commands.json", "training-input-audit.json"):
        require(logical in artifacts, f"{name}: missing required artifact {logical}")
    receipt = read_json(directory / "cloud-collection.json")
    require(receipt["completion_sha256"] == digest(completion_path) and receipt["metadata_paths_rewritten"] is False,
            f"{name}: collection receipt mismatch")
    config = read_json(directory / "train/config.json")
    metrics = read_json(directory / "train/metrics-val-original.json")
    manifest = read_json(directory / "manifest.json")
    history = read_json(directory / "train/history.json")
    validate_commands(read_json(directory / "commands.json"), name)
    for key, expected in {**FROZEN, "epochs": epochs, "train_samples": ARM[arm]["train_samples"], "seed": seed}.items():
        require(config.get(key) == expected, f"{name}: frozen setting {key} differs")
    loss = config["loss_details"]
    require(loss["main_head_weight"] == 1.0 and loss["auxiliary_head_weight"] == 0.4 and loss["dice"] is None
            and loss["head_objective"] == "CE" and loss["ce_reduction"] == "mean_over_target_class_weights",
            f"{name}: objective differs")
    samples, cases = validate_manifest(manifest, config, arm, name)
    input_audit = read_json(directory / "training-input-audit.json")
    require(input_audit["artifactType"] == "training_input_audit"
            and input_audit["manifestSha256"] == digest(directory / "manifest.json")
            and input_audit["splitCounts"] == {"train": ARM[arm]["train_samples"], "val": 75, "test": 74}
            and input_audit["splitCaseIds"] == cases and input_audit["classes"] == ONTOLOGY
            and input_audit["ignoreIndex"] == 255 and input_audit["ignoreSourceIds"] == [255], f"{name}: training-input audit disagrees")
    for split in ("train", "val", "test"):
        audit_rows = input_audit["samples"][split]
        require(record_digest(audit_rows) == input_audit["splitContentSha256"][split], f"{name}: input content fingerprint mismatch")
        metadata = [{k: v for k, v in row.items() if k not in ("imageSha256", "maskSha256")} for row in audit_rows]
        expected_metadata = [{k: v for k, v in row.items() if k not in PATH_FIELDS} for row in samples if row["split"] == split]
        require(metadata == expected_metadata, f"{name}: input content identities disagree with manifest")
        require(all(isinstance(row[key], str) and len(row[key]) == 64 and all(c in "0123456789abcdef" for c in row[key])
                    for row in audit_rows for key in ("imageSha256", "maskSha256")), f"{name}: invalid image/mask content hash")
    manifest_hash = canonical_digest(manifest)
    require(config["manifest_sha256"] == metrics["training_manifest_sha256"] == metrics["evaluation_manifest_sha256"] == manifest_hash,
            f"{name}: manifest identity mismatch")
    for key, expected in {"split": "val", "sample_count": 75, "case_count": 10, "limited_evaluation": False,
                          "limited_training": False, "input_size": FROZEN["input_size"], "ignore_index": 255,
                          "ignore_source_ids": [255], "device": "cuda", "architecture": FROZEN["architecture"],
                          "metric_resolution": {"basis": "original_mask_pixels", "sizes": [{"width": 854, "height": 480}]}}.items():
        require(metrics.get(key) == expected, f"{name}: evaluation {key} differs")
    frames = normalize_paths(metrics["frames"])
    require(len(frames) == 75 and metrics["evaluation_video_ids"] == cases["val"], f"{name}: evaluation identities differ")
    evaluated = [{"split": "val", "videoId": int(x["video_id"]), "frameNumber": x["frame_number"],
                  "timestampMs": x["timestamp_ms"], "imagePath": x["image_path"], "maskPath": x["mask_path"]} for x in frames]
    require(evaluated == [x for x in samples if x["split"] == "val"], f"{name}: evaluated frames differ from ordered validation samples")
    require([x["epoch"] for x in history] == list(range(1, epochs + 1)), f"{name}: incomplete or reordered history")
    for row in history:
        require(row["train_batches"] == math.ceil(ARM[arm]["train_samples"] / 2)
                and row["train_scored_pixels"] == config["class_weighting_details"]["scored_pixels"]
                and row["train_ignored_pixels"] == config["class_weighting_details"]["ignored_pixels"],
                f"{name}: actual training steps/pixel budget differs")
        require(not isinstance(row["duration_seconds"], bool) and isinstance(row["duration_seconds"], (int, float))
                and math.isfinite(row["duration_seconds"]) and row["duration_seconds"] > 0, f"{name}: invalid epoch duration")
    selected = max(history, key=lambda row: score(row["validation"]["foreground_macro_iou"], "selection IoU"))["epoch"]
    require(metrics["model_version"] == f"{config['run_id']}-epoch-{selected}", f"{name}: selected model version differs")
    require(metrics["checkpoint_sha256"] == artifacts["train/best.pt"]["sha256"], f"{name}: scored checkpoint differs")
    verify_confusion(metrics, name)

    job_path = Path(experiment["config"])
    require(experiment["config_sha256"] == digest(job_path), f"{name}: launch job config hash mismatch")
    job = read_json(job_path)
    container = job["workerPoolSpecs"][0]["containerSpec"]
    env = {x["name"]: x["value"] for x in container["env"]}
    require(len(env) == len(container["env"]), f"{name}: duplicate submitted environment variable")
    expected_env = {"HOLOSPEX_SOURCE_SHA256": bundles["source"]["sha256"], "HOLOSPEX_DATA_SHA256": bundles["data"]["sha256"],
                    "HOLOSPEX_LOSS": "ce", "HOLOSPEX_SEED": str(seed), "HOLOSPEX_EPOCHS": str(epochs), "HOLOSPEX_RUN_NAME": name}
    if arm == "added":
        expected_env.update({"HOLOSPEX_PREPARED_SHA256": bundles["prepared"]["sha256"], "HOLOSPEX_PREPARED_URI": bundles["prepared"]["uri"]})
        require("prepared-input-verification.json" in artifacts, f"{name}: missing prepared input verification")
        verification = read_json(directory / "prepared-input-verification.json")
        require(verification["verified"] is True and verification["originalSamplesPreserved"] is True
                and verification["heldOutSamplesPreserved"] is True and verification["addedTrainImageCount"] == 18
                and verification["counts"] == {"train": 361, "val": 75, "test": 74}, f"{name}: prepared input verification disagrees")
        for field, logical in (("packageSha256", "prepared-package.json"), ("sourceManifestSha256", "manifest-source.json"),
                               ("baseManifestSha256", "base-manifest-source.json")):
            require(logical in artifacts and verification[field] == digest(directory / logical), f"{name}: prepared source snapshot mismatch")
        require(verification["packageSha256"] == bundles["prepared"]["package_sha256"]
                and verification["sourceManifestSha256"] == bundles["prepared"]["prepared_manifest_sha256"],
                f"{name}: prepared sources differ from launch")
        verify_prepared_content(directory, manifest, input_audit, verification, name)
    else:
        require(not env.get("HOLOSPEX_PREPARED_SHA256") and not env.get("HOLOSPEX_PREPARED_URI"), f"{name}: base received prepared data")
    for key, expected in expected_env.items():
        require(env.get(key) == expected, f"{name}: submitted {key} differs")
    values = {}
    for label, keys in METRIC_PATHS.items():
        value = metrics
        for key in keys:
            value = value[key]
        values[label] = score(value, label)
    per_class = {}
    for basis, records in (("pooled", metrics["per_class"]), ("case_equal", metrics["case_equal"]["per_class"])):
        classes = {x["structureId"]: x for x in records}
        require(len(records) == len(ONTOLOGY) and set(classes) == {x["structureId"] for x in ONTOLOGY}, f"{name}: metric classes differ")
        per_class[basis] = {c: {k: score(classes[c][k], f"{basis} {c} {k}", nullable=k in ("precision", "recall")) for k in CLASS_METRICS}
                            for c in FOREGROUND}
    for label, basis, ids, key in (("small_pooled_iou", "pooled", SMALL, "iou"), ("small_case_equal_iou", "case_equal", SMALL, "iou"),
                                    ("foreground_iou", "pooled", FOREGROUND, "iou"), ("foreground_dice", "pooled", FOREGROUND, "dice")):
        require(math.isclose(values[label], statistics.mean(per_class[basis][c][key] for c in ids), abs_tol=1e-12),
                f"{name}: aggregate disagrees with per-class scores")
    row = {"run": name, "seed": seed, "arm": arm, "epochs": epochs, "selected_epoch": selected,
           "train_samples": ARM[arm]["train_samples"], "train_cases": ARM[arm]["train_cases"],
           "batches_per_epoch": math.ceil(ARM[arm]["train_samples"] / 2), "metrics": values, "per_class": per_class,
           "class_weights": config["class_weights"], "training_pixel_counts": config["class_weighting_details"]["counts"],
           "checkpoint_sha256": metrics["checkpoint_sha256"], "model_version": metrics["model_version"],
           "attempt": completion["attempt"], "job_name": experiment["job_name"], "output_uri": experiment["output_uri"],
           "completion_sha256": digest(completion_path), "config_sha256": digest(directory / "train/config.json"),
           "metrics_sha256": digest(directory / "train/metrics-val-original.json"), "job_config_sha256": digest(job_path),
           "manifest_sha256": manifest_hash, "training_input_audit_sha256": digest(directory / "training-input-audit.json"),
           "split_content_sha256": input_audit["splitContentSha256"], "all_artifact_hashes_verified": True,
           "pooled_and_case_equal_scores_recomputed_from_confusion": True,
           "recorded_optimizer_steps": sum(x["train_batches"] for x in history),
           "training_epoch_seconds": {"sum": sum(x["duration_seconds"] for x in history),
                                      "mean": statistics.mean(x["duration_seconds"] for x in history)},
           "cloud_started_at": completion.get("started_at"), "cloud_finished_at": completion.get("finished_at")}
    row["planned_optimizer_steps"] = row["batches_per_epoch"] * epochs
    audit = {"shared_config": shared_config(config), "frames": frames, "metric_policy": metrics["metric_policy"],
             "runtime": read_json(directory / "environment.json"), "container": container["imageUri"],
             "machine_spec": job["workerPoolSpecs"][0]["machineSpec"]}
    return row, audit, samples, input_audit["samples"]


def distribution(values):
    """Do not silently average over fewer seeds when a precision is undefined."""
    defined = [x for x in values if x is not None]
    require(all(math.isfinite(x) for x in defined), "Cannot average nonfinite seed results")
    complete = len(defined) == len(values)
    return {"mean": statistics.mean(defined) if complete else None, "min": min(defined) if complete else None,
            "max": max(defined) if complete else None, "n": len(values), "defined_n": len(defined)}


def difference(before, after):
    return None if before is None or after is None else 100 * (after - before)


def summarize(results_root, launch_manifest):
    launch = read_json(launch_manifest)
    experiments = {x["name"]: x for x in launch["experiments"]}
    require(len(launch["experiments"]) == 6 and set(experiments) == {x[0] for x in RUNS}, "Launch must contain exactly the frozen six runs")
    rows, common, samples_by_arm, contents_by_arm = [], None, {}, {}
    for name, seed, arm in RUNS:
        experiment = experiments[name]
        require(experiment["seed"] == seed and experiment["arm"] == arm and experiment["epochs"] == ARM[arm]["epochs"], f"{name}: launch seed/arm/budget differs")
        row, audit, samples, contents = load_run(results_root, name, seed, arm, experiment, launch["bundles"])
        require(common is None or audit == common, f"{name}: shared settings, runtime, policy, or ordered frames differ")
        require(arm not in samples_by_arm or samples_by_arm[arm] == samples, f"{name}: dataset changed within an arm")
        require(arm not in contents_by_arm or contents_by_arm[arm] == contents, f"{name}: image/mask bytes changed within an arm")
        samples_by_arm[arm], common = samples, audit
        contents_by_arm[arm] = contents
        rows.append(row)
    base, added = samples_by_arm["base"], samples_by_arm["added"]
    base_ids = {(str(x["videoId"]), x["frameNumber"]) for x in base}
    original = [x for x in added if (str(x["videoId"]), x["frameNumber"]) in base_ids]
    new = [x for x in added if (str(x["videoId"]), x["frameNumber"]) not in base_ids]
    require(original == base, "Added arm changed original sample metadata, split, paths, or ordering")
    for split in ("train", "val", "test"):
        original_content = [x for x in contents_by_arm["added"][split] if (str(x["videoId"]), x["frameNumber"]) in base_ids]
        require(original_content == contents_by_arm["base"][split], f"Added arm changed original {split} image/mask contents")
    require(len(new) == 18 and all(x["split"] == "train" and x.get("annotationSource") == "reviewed_partial_anatomy" for x in new),
            "Added arm must contain exactly 18 new reviewed partial train samples")
    require(len({str(x["videoId"]) for x in new}) == 18 and not {str(x["videoId"]) for x in new} & {str(x["videoId"]) for x in base},
            "Added images must come from 18 new training cases")
    # The same arm's weights and pixel counts must not vary between seeds.
    for arm in ARM:
        arm_rows = [r for r in rows if r["arm"] == arm]
        require(all((r["class_weights"], r["training_pixel_counts"]) == (arm_rows[0]["class_weights"], arm_rows[0]["training_pixel_counts"]) for r in arm_rows),
                f"{arm}: training pixel counts or weights changed across seeds")
    pairs = []
    for i in (0, 2, 4):
        before, after = rows[i:i + 2]
        pairs.append({"seed": before["seed"], "base_run": before["run"], "added_run": after["run"],
                      "added_minus_base_percentage_points": {key: difference(before["metrics"][key], after["metrics"][key]) for key in METRIC_PATHS},
                      "per_class_added_minus_base_percentage_points": {
                          basis: {c: {key: difference(before["per_class"][basis][c][key], after["per_class"][basis][c][key])
                                      for key in CLASS_METRICS} for c in FOREGROUND} for basis in ("pooled", "case_equal")}})
    groups = {arm: {key: distribution([r["metrics"][key] for r in rows if r["arm"] == arm]) for key in METRIC_PATHS} for arm in ARM}
    class_groups = {arm: {basis: {c: {key: distribution([r["per_class"][basis][c][key] for r in rows if r["arm"] == arm])
                                         for key in CLASS_METRICS} for c in FOREGROUND} for basis in ("pooled", "case_equal")} for arm in ARM}
    delta_summary = {key: distribution([p["added_minus_base_percentage_points"][key] for p in pairs]) for key in METRIC_PATHS}
    positive_pairs = sum(all(pair["added_minus_base_percentage_points"][key] > 0 for key in ("small_pooled_iou", "small_case_equal_iou")) for pair in pairs)
    return {"report_type": "reviewed_partial_data_three_paired_seed_validation_comparison", "generated_at": datetime.now(timezone.utc).isoformat(),
            "launch_manifest_sha256": digest(launch_manifest), "bundle_sha256": {key: launch["bundles"][key]["sha256"] for key in ("source", "data", "prepared")},
            "runs": rows, "paired_seeds": pairs, "group_summary_fraction_units": groups,
            "per_class_group_summary_fraction_units": class_groups,
            "paired_delta_summary_percentage_points": delta_summary,
            "per_class_paired_delta_summary_percentage_points": {
                basis: {c: {key: distribution([p["per_class_added_minus_base_percentage_points"][basis][c][key] for p in pairs])
                            for key in CLASS_METRICS} for c in FOREGROUND} for basis in ("pooled", "case_equal")},
            "compute_budget": {"base": {"epochs": 42, "batches_per_epoch": 172, "planned_optimizer_steps": 7224},
                               "added": {"epochs": 40, "batches_per_epoch": 181, "planned_optimizer_steps": 7240},
                               "added_step_increase_percent": 100 * (7240 / 7224 - 1),
                               "added_batches_per_epoch_increase_percent": 100 * (181 / 172 - 1),
                               "selection_caveat": "Base has 42 validation selection opportunities and added has 40; epoch budgets were frozen before results."},
            "audit": {"all_six_complete": True, "all_shared_settings_match": True, "original_samples_and_heldouts_unchanged_after_path_normalization": True,
                      "original_sample_image_and_mask_hashes_match": True,
                      "pooled_and_case_equal_scores_recomputed_from_confusion": True,
                      "new_train_images": 18, "new_train_cases": 18, "recorded_evaluation_uses_validation_only": True,
                      "shared_config_sha256": canonical_digest(common["shared_config"]),
                      "normalized_heldout_samples_sha256": canonical_digest([x for x in base if x["split"] != "train"]),
                      "container": common["container"], "machine_spec": common["machine_spec"], "runtime": common["runtime"], "metric_policy": common["metric_policy"]},
            "practical_pilot_criterion": {"minimum_mean_gain_percentage_points_both_small_metrics": 1.0,
                                          "minimum_pairs_positive_on_both_small_metrics": 2,
                                          "pairs_positive_on_both_small_metrics": positive_pairs,
                                          "met": positive_pairs >= 2 and all(delta_summary[key]["mean"] >= 1.0 for key in ("small_pooled_iou", "small_case_equal_iou")),
                                          "requires_review": "Inspect class regressions and precision/recall before labeling recommendation; no automatic demo checkpoint change."},
            "interpretation": "Three paired seeds with approximately matched optimizer steps; validation reused, CUDA nondeterministic. Added data also changes class weights and shuffle order. No significance, clinical-validation, or checkpoint-promotion claim. Undefined per-class seed values make the three-seed aggregate undefined. All reported scores are original-grid validation; this script performs no evaluation."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        encoded = json.dumps(summarize(args.results_root, args.launch_manifest), indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                output.write(encoded)
        else:
            print(encoded, end="")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f"Reviewed-data comparison failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
