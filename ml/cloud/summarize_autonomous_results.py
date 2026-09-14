"""Create a fresh local report by re-auditing collected autonomous experiments.

This reads a snapshot of controller state and saved artifacts. It does not query
cloud services, load model tensors, run inference, or modify experiment inputs.
All eligible scores are recomputed by the existing native-validation auditor.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from audit_autonomous_results import TARGET_IOU, _read_json, _sha256, summarize_run


TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
FAILURES = TERMINAL - {"JOB_STATE_SUCCEEDED"}
MAX_ENTRIES = 256
SCORE_KEYS = ("foreground_macro_iou", "small_pooled_iou", "small_case_equal_iou")


def _timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Report timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def _path(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("Artifact paths must be nonempty strings")
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _result_root(run, base):
    location = run.get("result_dir") or run.get("audit", {}).get("run_root")
    return _path(location, base) if location else base / "results" / run["name"]


def _source(run, base):
    """Keep source lineage separate from the independently verified metrics."""
    recorded = run.get("bundle_references", {}).get("source", {}).get("sha256")
    result = {"sha256": recorded, "basis": "controller_record_only" if recorded else "unavailable"}
    if not run.get("config"):
        return result
    try:
        path = _path(run["config"], base)
        if _sha256(path) != run.get("config_sha256"):
            raise ValueError("Submitted job configuration SHA-256 differs from the controller record")
        config = _read_json(path)
        environment = config["workerPoolSpecs"][0]["containerSpec"]["env"]
        values = {item["name"]: item["value"] for item in environment}
        if len(values) != len(environment):
            raise ValueError("Submitted job configuration has duplicate environment entries")
        actual = values.get("HOLOSPEX_SOURCE_SHA256")
        if not isinstance(actual, str) or re.fullmatch(r"[0-9a-f]{64}", actual) is None:
            raise ValueError("Submitted job configuration has no valid source SHA-256")
        if recorded is not None and recorded != actual:
            raise ValueError("Submitted job source SHA-256 differs from the controller bundle record")
        result.update(sha256=actual, basis="submitted_job_config_sha256_checked", config_path=str(path),
                      config_sha256=run["config_sha256"])
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result["warning"] = str(error)
    return result


def _worker_timing(run, base, snapshot):
    path = base / "jobs" / (run["name"] + ".status.json")
    result = {"vertex_worker_runtime_seconds": None, "vertex_worker_elapsed_seconds": None,
              "basis": "saved_Vertex_startTime_to_endTime; includes_bootstrap_preparation_training_evaluation_upload; excludes_queue"}
    if not path.is_file():
        return result
    try:
        status = _read_json(path)
        result.update(status_path=str(path), saved_vertex_state=status.get("state"),
                      saved_update_time=status.get("updateTime"), worker_start_time=status.get("startTime"),
                      worker_end_time=status.get("endTime"))
        if status.get("startTime"):
            start = _timestamp(status["startTime"])
            if status.get("endTime"):
                duration = (_timestamp(status["endTime"]) - start).total_seconds()
                if duration < 0:
                    raise ValueError("Saved Vertex endTime precedes startTime")
                result["vertex_worker_runtime_seconds"] = duration
            elif status.get("state") not in TERMINAL:
                result["vertex_worker_elapsed_seconds"] = max(0.0, (snapshot - start).total_seconds())
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["warning"] = str(error)
    return result


def _recipe(config):
    keys = ("architecture", "initialization", "seed", "input_size", "batch_size", "learning_rate",
            "lr_schedule", "warmup_epochs", "backbone_lr_multiplier", "weight_decay", "augmentation",
            "sampling", "loss", "dice_weight", "lovasz_weight", "max_duration_seconds", "class_weighting")
    result = {key: config.get(key) for key in keys}
    result["auxiliary_loss_weight"] = config.get("auxiliary_loss_weight", config.get("loss_details", {}).get("auxiliary_head_weight"))
    result["initial_checkpoint"] = config.get("initial_checkpoint")
    result["backbone_checkpoint"] = config.get("backbone_checkpoint")
    result["dataset"] = config.get("dataset", {}).get("dataset")
    return result


def _summarize_entry(run, base, snapshot, historical):
    result = {"name": run["name"], "state": run.get("state", "unknown"), "historical": historical,
              "verified": False, "verification_status": "not_successful", "error": run.get("error"),
              "inspection_error": run.get("inspection_error"), "job_name": run.get("job_name"),
              "submitted_at": run.get("submitted_at"), "requested_recipe": run.get("recipe", {}),
              "selected_epoch": None, "epochs_completed": None, "epochs_requested": run.get("recipe", {}).get("epochs"),
              "result_dir": str(_result_root(run, base)),
              "source_bundle": _source(run, base), "timing": _worker_timing(run, base, snapshot)}
    if result["state"] != "JOB_STATE_SUCCEEDED":
        return result
    try:
        root = _result_root(run, base)
        result["result_dir"] = str(root)
        if not root.is_dir():
            result["verification_status"] = "awaiting_local_result"
            return result
        # Cached audit booleans and scores never enter this decision.
        verified = summarize_run(root)
        config = _read_json(root / "train/config.json")
        history = _read_json(root / "train/history.json")
        completion = _read_json(root / "cloud-completion.json")
        result.update({key: verified[key] for key in (*SCORE_KEYS, "selected_epoch", "epochs_completed",
                       "epochs_requested", "checkpoint", "checkpoint_sha256", "per_class", "duration_limited")})
        result.update(verified=True, verification_status="independently_reaudited", recipe=_recipe(config),
                      metric_basis={"split": "val", "native_size": {"width": 854, "height": 480},
                                    "sample_count": 75, "case_count": 10, "foreground_class_count": 6},
                      audit=verified["audit"], target_reached=verified["foreground_macro_iou"] >= TARGET_IOU,
                      train_video_ids=verified["train_video_ids"])
        durations = [row.get("duration_seconds") for row in history]
        valid = all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in durations)
        call_duration = completion.get("training_duration_seconds")
        valid_call_duration = type(call_duration) in (int, float) and math.isfinite(call_duration) and call_duration >= 0
        result["timing"].update(training_epoch_seconds=sum(durations) if valid else None,
                                 training_epoch_time_basis="sum_of_train_plus_per_epoch_validation; excludes_setup_and_checkpoint_writes",
                                 training_call_seconds=call_duration if valid_call_duration else None,
                                 training_call_time_basis="trainer_reported_wall_time_including_setup_and_checkpoint_writes")
    except (OSError, ValueError, TypeError, KeyError) as error:
        # Even a previously successful cached audit is excluded if local evidence
        # is now missing, inconsistent, corrupt, or no longer comparable.
        result.update(verified=False, verification_status="audit_failed", verification_error=str(error))
        for key in (*SCORE_KEYS, "target_reached", "per_class"):
            result.pop(key, None)
    return result


def _best(runs):
    eligible = [row for row in runs if row.get("verified") is True]
    return max(eligible, key=lambda row: row["foreground_macro_iou"], default=None)


def summarize_state(state_path, *, snapshot_time=None):
    state_path = Path(state_path).resolve()
    snapshot = snapshot_time or datetime.now(timezone.utc)
    if snapshot.tzinfo is None:
        raise ValueError("Snapshot time must include a timezone")
    before = state_path.read_bytes()
    state = _read_json(state_path)
    if before != state_path.read_bytes():
        raise ValueError("Controller state changed while taking the snapshot; retry reporting")
    if not isinstance(state, dict):
        raise ValueError("Controller state must be an object")
    runs, references = state.get("runs", []), state.get("reference_runs", [])
    if not isinstance(runs, list) or not isinstance(references, list) or len(runs) + len(references) > MAX_ENTRIES:
        raise ValueError(f"Report requires lists totaling at most {MAX_ENTRIES} runs and references")
    names = []
    for run in runs + references:
        if not isinstance(run, dict) or not isinstance(run.get("name"), str) or run["name"] in {".", ".."} or re.fullmatch(r"[A-Za-z0-9._-]+", run["name"]) is None:
            raise ValueError("Every run requires a simple nonempty name")
        names.append(run["name"])
    if len(names) != len(set(names)):
        raise ValueError("Run/reference names must be unique")
    configured = state.get("target", {})
    if configured and (configured.get("metric") != "foreground_macro_iou" or configured.get("threshold") != TARGET_IOU
                       or configured.get("split") != "val"):
        raise ValueError("Controller target differs from the fixed native-validation audit target")
    deadline = _timestamp(state["deadline"])
    rows = [_summarize_entry(run, state_path.parent, snapshot, False) for run in runs]
    reference_rows = [_summarize_entry(run, state_path.parent, snapshot, True) for run in references]
    best, reference = _best(rows), _best(reference_rows)
    available = _best(rows + reference_rows)
    tradeoffs = []
    if best:
        for row in best["per_class"][1:]:
            previous = next((item for item in reference["per_class"] if item["structureId"] == row["structureId"]), None) if reference else None
            old = previous["iou"] if previous else None
            tradeoffs.append({"structure_id": row["structureId"], "iou": row["iou"], "reference_iou": old,
                              "delta_percentage_points": 100 * (row["iou"] - old) if old is not None and row["iou"] is not None else None})
    active = sum(row["state"] not in TERMINAL for row in rows)
    target_met = any(row.get("verified") and row["foreground_macro_iou"] >= TARGET_IOU for row in rows)
    return {"format_version": 1, "artifact_type": "autonomous_training_results_snapshot",
            "snapshot_at": snapshot.astimezone(timezone.utc).isoformat(), "state_path": str(state_path),
            "state_sha256": hashlib.sha256(before).hexdigest(), "state_updated_at": state.get("updated_at"),
            "snapshot_scope": "local_controller_state_and_saved_artifacts_only; no_cloud_refresh",
            "controller_status": state.get("status", "unknown"), "controller_stop_reason": state.get("stop_reason"),
            "search_active": state.get("status") in {"ready", "running", "stopping"} or active > 0,
            "started_at": state.get("started_at"), "deadline": deadline.isoformat(), "deadline_reached": snapshot >= deadline,
            "target": {"metric": "six_class_foreground_macro_iou", "threshold": TARGET_IOU, "split": "val", "samples": 75,
                       "cases": 10, "resolution": "original_854x480", "met_by_verified_search_run": target_met},
            "counts": {"runs": len(rows), "active": active, "successful_state": sum(row["state"] == "JOB_STATE_SUCCEEDED" for row in rows),
                       "verified": sum(row["verified"] for row in rows), "failed_or_cancelled_or_expired": sum(row["state"] in FAILURES for row in rows),
                       "audit_failed": sum(row["verification_status"] == "audit_failed" for row in rows),
                       "awaiting_local_result": sum(row["verification_status"] == "awaiting_local_result" for row in rows),
                       "queued_recipes": len(state.get("queue", [])), "references": len(reference_rows),
                       "verified_references": sum(row["verified"] for row in reference_rows), "by_state": dict(Counter(row["state"] for row in rows))},
            "runs": rows, "references": reference_rows, "best_search_run": best["name"] if best else None,
            "best_available_run": available["name"] if available else None, "comparison_reference": reference["name"] if reference else None,
            "best_search_per_class_tradeoffs": tradeoffs,
            "interpretation": "Validation comparisons guide this search. Reused validation results do not establish statistical significance or clinical validity. No test-set evaluation was performed."}


def _cell(value):
    return str(value if value is not None else "unavailable").replace("|", "\\|").replace("\n", " ").replace("<", "&lt;")


def _percent(value):
    return f"{100 * value:.2f}%" if value is not None else "—"


def render_markdown(summary):
    counts = summary["counts"]
    lines = ["# Autonomous training results", "", f"Snapshot: {summary['snapshot_at']}. Controller state last updated: {_cell(summary['state_updated_at'])}.",
             f"Search active: **{str(summary['search_active']).lower()}**. Controller status: {_cell(summary['controller_status'])}.",
             f"Deadline: {summary['deadline']}; reached at snapshot: **{str(summary['deadline_reached']).lower()}**.",
             f"75% target met by an independently re-audited search run: **{str(summary['target']['met_by_verified_search_run']).lower()}**.", "",
             "All scored rows use the fixed 75 validation frames, ten cases, six foreground classes, and original 854 × 480 masks. Small anatomy covers duct, artery, plate, and triangle. This snapshot reads local saved evidence; it does not refresh cloud status.", "",
             f"Runs: {counts['runs']}; independently verified: {counts['verified']}; active: {counts['active']}; failed/cancelled/expired: {counts['failed_or_cancelled_or_expired']}; audit failures: {counts['audit_failed']}; successful runs awaiting local results: {counts['awaiting_local_result']}; queued recipes: {counts['queued_recipes']}.", "",
             f"Best search run: **{_cell(summary['best_search_run'])}**. Best available including references: **{_cell(summary['best_available_run'])}**.", "",
             "| Run | Saved state / verification | Foreground IoU | Small pooled IoU | Small case-equal IoU | Selected / completed / requested epochs |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summary["runs"] + summary["references"]:
        label = row["name"] + (" (reference)" if row["historical"] else "")
        epochs = " / ".join(str(row[key]) if row[key] is not None else "—" for key in ("selected_epoch", "epochs_completed", "epochs_requested"))
        lines.append(f"| {_cell(label)} | {_cell(row['state'])} / {_cell(row['verification_status'])} | " + " | ".join(_percent(row.get(key)) for key in SCORE_KEYS) + f" | {epochs} |")
    lines += ["", "## Best search run versus reviewed reference", "",
              f"Reference: {_cell(summary['comparison_reference'])}. Deltas are percentage points on the same audited validation set.", "",
              "| Structure | Best search IoU | Reference IoU | Delta (pp) |", "| --- | ---: | ---: | ---: |"]
    for row in summary["best_search_per_class_tradeoffs"]:
        delta = row["delta_percentage_points"]
        lines.append(f"| {_cell(row['structure_id'])} | {_percent(row['iou'])} | {_percent(row['reference_iou'])} | {f'{delta:+.2f}' if delta is not None else '—'} |")
    lines += ["", "## Configuration, provenance, and timing", "",
              "Epoch time sums training plus per-epoch validation. Vertex worker runtime includes bootstrap, preparation, training, evaluation, and upload; it excludes queue time. Missing timing is left unavailable."]
    for row in summary["runs"] + summary["references"]:
        lines += ["", f"### {_cell(row['name'])}", ""]
        if row["verified"]:
            recipe = row["recipe"]
            lines.append(f"Architecture: {_cell(recipe['architecture'])}; initialization: {_cell(recipe['initialization'])}; input: {_cell(recipe['input_size'])}.")
            keys = ("seed", "batch_size", "learning_rate", "lr_schedule", "warmup_epochs", "backbone_lr_multiplier", "weight_decay", "augmentation", "sampling", "loss", "dice_weight", "lovasz_weight", "auxiliary_loss_weight")
            lines.append("Recipe: " + "; ".join(f"{key}={_cell(recipe.get(key))}" for key in keys) + ".")
            lines.append(f"Checkpoint SHA-256: `{row['checkpoint_sha256']}`. Local checkpoint: `{_cell(row['checkpoint'])}`.")
            for key in ("initial_checkpoint", "backbone_checkpoint"):
                if recipe.get(key):
                    lines.append(f"{key}: `{_cell(json.dumps(recipe[key], sort_keys=True))}`.")
        else:
            lines.append("Requested recipe (execution not verified): " + _cell(json.dumps(row["requested_recipe"], sort_keys=True)) + ".")
        source, timing = row["source_bundle"], row["timing"]
        lines.append(f"Source bundle SHA-256: `{_cell(source['sha256'])}`; evidence: {_cell(source['basis'])}.")
        lines.append(f"Epoch time: {_cell(timing.get('training_epoch_seconds'))} s; full training-call time: {_cell(timing.get('training_call_seconds'))} s; Vertex worker runtime: {_cell(timing.get('vertex_worker_runtime_seconds'))} s; active worker elapsed time: {_cell(timing.get('vertex_worker_elapsed_seconds'))} s.")
        for key in ("error", "inspection_error", "verification_error"):
            if row.get(key):
                lines.append(f"{key}: {_cell(row[key])}.")
        for metadata in (source, timing):
            if metadata.get("warning"):
                lines.append("Metadata warning: " + _cell(metadata["warning"]) + ".")
    lines += ["", summary["interpretation"], "", f"State snapshot SHA-256: `{summary['state_sha256']}`.", ""]
    return "\n".join(lines)


def create_report(state_path, output_dir, *, snapshot_time=None):
    state_path, output = Path(state_path).resolve(), Path(output_dir).resolve()
    summary = summarize_state(state_path, snapshot_time=snapshot_time)
    protected = [state_path.parent / "jobs", state_path.parent / "results"]
    protected.extend(Path(row["result_dir"]) for row in summary["runs"] + summary["references"] if row.get("result_dir"))
    if output.exists() or any(output == root or output.is_relative_to(root) for root in protected):
        raise ValueError("Report output must be a fresh directory outside existing run artifacts and job status inputs")
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    markdown = render_markdown(summary)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x") as destination:
        destination.write(payload)
    with (output / "RESULTS.md").open("x") as destination:
        destination.write(markdown)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = create_report(args.state, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "snapshot_at": summary["snapshot_at"],
                      "verified_runs": summary["counts"]["verified"], "target_met": summary["target"]["met_by_verified_search_run"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
