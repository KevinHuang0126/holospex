"""Bounded adaptive Vertex experiments using existing immutable data bundles.

Run only when cloud training is authorized. This controller owns only the jobs
recorded in its state file, never project-wide jobs. A persisted submit intent
prevents an ambiguous network result from being submitted twice.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

from collect_vertex_results import collect
from audit_autonomous_results import summarize_run

PROJECT = "eastwest72hack26bos-501"
REGION = "us-central1"
BUCKET = f"gs://{PROJECT}-holospex-ml"
GCLOUD = ["gcloud", "--configuration=holospex"]
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
MOBILE = "deeplabv3_mobilenet_v3_large"
DETAIL = "deeplabv3plus_mobilenet_v3_large"
MAX_LAUNCHES = 16
MAX_PARALLEL = 2
MIN_LAUNCH_SECONDS = 480
RESUMABLE = {"ready", "running", "stopping"}
TARGET_IOU = 0.75
SHA256 = re.compile(r"[0-9a-f]{64}")
SOURCE_ENV = {"HOLOSPEX_SOURCE_URI", "HOLOSPEX_SOURCE_SHA256"}
COLLECTION_TIMEOUT_SECONDS = 600
COLLECTION_DEADLINE_RESERVE_SECONDS = 30
FINAL_COLLECTION_TIMEOUT_SECONDS = 120


def now():
    return datetime.now(timezone.utc)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command(args, *, timeout=120):
    result = subprocess.run(GCLOUD + args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"gcloud {args[0:3]}: {result.stderr[-3000:]}")
    return result.stdout


def recipe(label, reason, **changes):
    return {"label": label, "reason": reason, "architecture": MOBILE,
            "epochs": 80, "width": 672, "height": 384, "lr": 0.0003,
            "lr_schedule": "cosine", "warmup_epochs": 0,
            "backbone_lr_multiplier": 1.0, "weight_decay": 0.01,
            "augmentation": "none", "sampling": "uniform", "loss": "ce",
            "dice_weight": 0.0, "seed": 42, **changes}


def initial_recipes(initial_checkpoint):
    return [
        recipe("warm-cosine", "Constant-rate reviewed baseline overfits; test gentle weight-only fine-tuning.",
               epochs=40, lr=0.00003, initial_checkpoint=initial_checkpoint),
        recipe("reviewed-896-cosine", "Combine reviewed supervision with detail-preserving resolution and decay.",
               width=896, height=512),
        recipe("detail-672-cosine", "Test stride-4 low-level feature decoder for thin-structure boundaries.",
               architecture=DETAIL, backbone_lr_multiplier=0.1, warmup_epochs=3),
        recipe("low-lr-cosine", "Test smaller initial learning rate against noisy constant-rate plateau.", lr=0.0001),
    ]


def _auxiliary_loss_weight(candidate):
    value = candidate.get("auxiliary_loss_weight", 0.4)
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10:
        raise ValueError("Recipe auxiliary_loss_weight must be finite and between 0 and 10")
    return value


def validate_recipe(candidate):
    """Accept bounded training options, never commands or data/split overrides."""
    if isinstance(candidate, dict) and "replicate_best_full" in candidate:
        if set(candidate) != {"replicate_best_full", "label", "reason", "seed"}:
            raise ValueError("Replication marker permits only group, label, reason, and seed")
        for key in ("replicate_best_full", "label"):
            if not isinstance(candidate[key], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", candidate[key]):
                raise ValueError("Replication group and label must be short lowercase identifiers")
        if not isinstance(candidate["reason"], str) or not candidate["reason"].strip():
            raise ValueError("Replication marker reason is required")
        if type(candidate["seed"]) is not int or candidate["seed"] not in {43, 44}:
            raise ValueError("Replication marker seed must be 43 or 44")
        return
    required = set(recipe("", ""))
    if not isinstance(candidate, dict) or set(candidate) - required - {"initial_checkpoint", "backbone_checkpoint", "lovasz_weight", "auxiliary_loss_weight"}:
        raise ValueError("Queue recipes contain unsupported fields")
    if required - set(candidate):
        raise ValueError("Queue recipes must include all standard recipe fields")
    if not isinstance(candidate["label"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", candidate["label"]):
        raise ValueError("Recipe label must be a short lowercase identifier")
    if not isinstance(candidate["reason"], str) or not candidate["reason"].strip():
        raise ValueError("Recipe reason is required")
    choices = {"architecture": {MOBILE, DETAIL, "deeplabv3_resnet50"}, "lr_schedule": {"none", "cosine"},
               "augmentation": {"none", "mild"}, "sampling": {"uniform", "case_balanced"},
               "loss": {"ce", "ce_generalized_dice", "ce_lovasz"}}
    for key, values in choices.items():
        if not isinstance(candidate[key], str) or candidate[key] not in values:
            raise ValueError(f"Unsupported recipe {key}")
    for key, lower, upper in (("epochs", 1, 240), ("width", 128, 1536), ("height", 128, 1024),
                              ("warmup_epochs", 0, 239), ("seed", 0, 2**32 - 1)):
        if type(candidate[key]) is not int or not lower <= candidate[key] <= upper:
            raise ValueError(f"Recipe {key} must be an integer from {lower} through {upper}")
    if candidate["warmup_epochs"] >= candidate["epochs"]:
        raise ValueError("Recipe warmup_epochs must be less than epochs")
    for key, lower, upper in (("lr", 1e-8, 0.1), ("weight_decay", 0, 1),
                              ("backbone_lr_multiplier", 1e-6, 10), ("dice_weight", 0, 10),
                              ("lovasz_weight", 0, 10)):
        value = candidate.get(key, 0.25)
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"Recipe {key} must be finite and between {lower} and {upper}")
    _auxiliary_loss_weight(candidate)
    validate_initialization(candidate)


def validate_initialization(candidate):
    backbone = candidate.get("backbone_checkpoint")
    if backbone is not None:
        if candidate.get("architecture") != "deeplabv3_resnet50":
            raise ValueError("Backbone checkpoint is supported only for deeplabv3_resnet50")
        if candidate.get("initial_checkpoint") is not None:
            raise ValueError("Backbone checkpoint and initial checkpoint are mutually exclusive")
    for key in ("initial_checkpoint", "backbone_checkpoint"):
        checkpoint = candidate.get(key)
        if checkpoint is None:
            continue
        digest = checkpoint.get("sha256", "") if isinstance(checkpoint, dict) else ""
        uri = checkpoint.get("uri", "") if isinstance(checkpoint, dict) else ""
        namespaces = ("objects", "bundles") if key == "backbone_checkpoint" else ("objects",)
        if (not isinstance(digest, str) or not SHA256.fullmatch(digest) or not isinstance(uri, str)
                or not uri.startswith(BUCKET + "/") or not any(f"/{namespace}/{digest}/" in uri for namespace in namespaces)):
            raise ValueError(f"{key} must be an immutable object in the session bucket")


def template_environment(template):
    workers = template.get("workerPoolSpecs", [])
    if not isinstance(workers, list) or len(workers) != 1:
        raise ValueError("Template must contain exactly one A100 worker pool")
    worker = workers[0]
    if (type(worker.get("replicaCount")) not in (int, str) or worker.get("replicaCount") not in (1, "1")
            or worker.get("machineSpec") != {"machineType": "a2-highgpu-1g", "acceleratorType": "NVIDIA_TESLA_A100", "acceleratorCount": 1}
            or type(worker["machineSpec"]["acceleratorCount"]) is not int
            or template.get("scheduling", {}).get("strategy") != "SPOT"):
        raise ValueError("Template must retain a single A100 SPOT worker")
    container = worker["containerSpec"]
    if (container.get("command") != ["python", "-u", "-c"]
            or not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", container.get("imageUri", ""))):
        raise ValueError("Template must retain a digest-pinned image and the Python bootstrap command")
    entries = container.get("env", [])
    if not isinstance(entries, list) or any(not isinstance(entry, dict) or set(entry) != {"name", "value"}
                                          or not isinstance(entry["name"], str) or not isinstance(entry["value"], str)
                                          for entry in entries):
        raise ValueError("Template environment must contain string name/value pairs")
    environment = {entry["name"]: entry["value"] for entry in entries}
    if len(environment) != len(entries):
        raise ValueError("Duplicate template environment variable")
    if environment.get("HOLOSPEX_PROJECT") != PROJECT:
        raise ValueError("Template project cannot change")
    return environment


def verified_score(run):
    audit = run.get("audit", {})
    score = audit.get("foreground_macro_iou")
    if (run.get("state") == "JOB_STATE_SUCCEEDED" and audit.get("verified") is True
            and type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1):
        return score
    return None


def target_reached(runs):
    # The independently verified numerical score is authoritative; a cached
    # boolean alone must never stop the search or announce a successful model.
    return any((score := verified_score(run)) is not None and score >= TARGET_IOU for run in runs)


def best_run(runs):
    completed = [r for r in runs if verified_score(r) is not None]
    return max(completed, key=lambda r: r["audit"]["foreground_macro_iou"], default=None)


def best_full_run(runs):
    """Only an owned, audited, fully executed concrete recipe can seed a group."""
    eligible = []
    for run in runs:
        audit, candidate = run.get("audit", {}), run.get("recipe", {})
        if not isinstance(audit, dict) or not isinstance(candidate, dict):
            continue
        if (verified_score(run) is None or run.get("historical") is True
                or audit.get("duration_limited") is not False
                or type(audit.get("epochs_completed")) is not int or audit["epochs_completed"] <= 0
                or type(audit.get("epochs_requested")) is not int
                or audit["epochs_completed"] != audit["epochs_requested"]
                or candidate.get("epochs") != audit["epochs_requested"]
                or "replicate_best_full" in candidate
                or not isinstance(audit.get("checkpoint_sha256"), str)
                or not SHA256.fullmatch(audit["checkpoint_sha256"])):
            continue
        try:
            validate_recipe(candidate)
        except ValueError:
            continue
        eligible.append(run)
    return best_run(eligible)


def choose_next(state, remaining_seconds):
    """Use verified validation results; all proposals retain the held-out data."""
    if remaining_seconds < MIN_LAUNCH_SECONDS or len(state["runs"]) >= min(state.get("max_runs", MAX_LAUNCHES), MAX_LAUNCHES):
        return None
    if state["queue"]:
        candidate = state["queue"].pop(0)
    else:
        best = best_run([*state["runs"], *state.get("reference_runs", [])])
        if best is None:
            return None
        used = {r["recipe"]["label"] for r in state["runs"]}
        base = copy.deepcopy(best["recipe"])
        name = best["name"]
        options = [
            ("best-refine", {"epochs": 40, "lr": 0.00001, "lr_schedule": "cosine", "warmup_epochs": 0,
                             "initial_checkpoint": best["checkpoint_reference"], "backbone_checkpoint": None},
             f"Refine current verified leader {name} with a lower learning rate."),
            ("detail-896", {"architecture": DETAIL, "width": 896, "height": 512,
                            "epochs": 80, "lr": 0.0003, "backbone_lr_multiplier": 0.1, "warmup_epochs": 3,
                            "initial_checkpoint": None, "backbone_checkpoint": None},
             "Test whether decoder detail and the larger reviewed-data grid work together."),
            ("best-mild", {"augmentation": "mild", "epochs": 80},
             f"Test whether augmentation helps the scheduled recipe selected from {name}."),
            ("best-decay", {"weight_decay": 0.05, "epochs": 80},
             f"Test stronger regularization after the optimization comparison; reference {name}."),
            ("best-native", {"width": 854, "height": 480, "epochs": 80},
             f"Test exact native geometry without aspect-ratio distortion; reference {name}."),
        ]
        candidate = None
        for label, changes, reason in options:
            if label not in used:
                candidate = {**base, **changes, "label": label, "reason": reason}
                break
        if candidate is None:
            seed = 43 + sum(r["recipe"]["label"].startswith("replicate-") for r in state["runs"])
            candidate = {**base, "seed": seed, "label": f"replicate-{seed}",
                         "reason": f"Measure seed variability of verified leader {name}."}
    # Reduced budgets are explicit, with selection from completed epochs only.
    if remaining_seconds < 1500 and "replicate_best_full" not in candidate:
        candidate["epochs"] = min(candidate["epochs"], max(5, int((remaining_seconds - 360) / 35)))
        candidate["warmup_epochs"] = min(candidate["warmup_epochs"], candidate["epochs"] - 1)
        candidate["reason"] += " Epoch budget reduced to fit the remaining wall-clock window."
    return candidate


class Controller:
    def __init__(self, state_path):
        self.path = Path(state_path).resolve()
        self.root = self.path.parent
        self.state = json.loads(self.path.read_text())
        for key, default, upper in (("max_parallel", MAX_PARALLEL, MAX_PARALLEL), ("max_runs", MAX_LAUNCHES, MAX_LAUNCHES)):
            value = self.state.setdefault(key, default)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{key} must be an integer from 1 through {upper}")
        if type(self.state.get("stop_when_exhausted", False)) is not bool:
            raise ValueError("stop_when_exhausted must be a boolean")
        deadline = datetime.fromisoformat(self.state["deadline"])
        if deadline.tzinfo is None:
            raise ValueError("Controller deadline must include a timezone")

    def event(self, kind, **details):
        record = {"at": now().isoformat(), "event": kind, **details}
        with (self.root / "events.jsonl").open("a") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)

    def save(self):
        self.state["updated_at"] = now().isoformat()
        write_json(self.path, self.state)
        best = best_run([*self.state["runs"], *self.state.get("reference_runs", [])])
        rows = ["# Autonomous training status", "", f"Deadline: {self.state['deadline']}",
                f"State: {self.state['status']}", "",
                "Target: 75% six-class foreground mean IoU on the fixed 75 original-grid validation frames.", "",
                "| Run | State | Native foreground IoU | Selected epoch |", "| --- | --- | ---: | ---: |"]
        for run in self.state["runs"]:
            audit = run.get("audit", {})
            validated = verified_score(run)
            score = f"{validated * 100:.2f}%" if validated is not None else "pending"
            rows.append(f"| {run['name']} | {run.get('state', 'planned')} | {score} | {audit.get('selected_epoch', '—')} |")
        if best:
            rows += ["", f"Current verified leader: {best['name']}.",
                     f"Checkpoint SHA-256: `{best['audit']['checkpoint_sha256']}`."]
        (self.root / "STATUS.md").write_text("\n".join(rows) + "\n")

    def remaining(self):
        return (datetime.fromisoformat(self.state["deadline"]) - now()).total_seconds()

    def batch_exhausted(self):
        """Optionally close a bounded batch after terminal results are inspected.

        Recorded collection errors still receive the normal bounded final audit;
        they are never treated as verified results or a successful target hit.
        """
        runs = self.state["runs"]
        return (self.state.get("stop_when_exhausted", False)
                and len(runs) >= self.state["max_runs"]
                and all(run["state"] in TERMINAL for run in runs)
                and all(run["state"] != "JOB_STATE_SUCCEEDED"
                        or verified_score(run) is not None
                        or ("audit" not in run and bool(run.get("inspection_error")))
                        for run in runs))

    def validate_template_proposal(self, template, receipt_path):
        """Only replace source code bound to a locally reviewed bundle receipt.

        The session directory is the trusted operator handoff. No archive is
        extracted, and no proposed code is executed by this controller.
        """
        before = template_environment(self.state["template"])
        after = template_environment(template)
        for kind in ("data", "prepared"):
            for field in ("uri", "sha256"):
                key = f"HOLOSPEX_{kind.upper()}_{field.upper()}"
                if after.get(key) != self.state[kind][field] or after.get(key) != before.get(key):
                    raise ValueError(f"Template {kind} identity cannot change")
        old_fixed, new_fixed = copy.deepcopy(self.state["template"]), copy.deepcopy(template)
        for fixed in (old_fixed, new_fixed):
            container = fixed["workerPoolSpecs"][0]["containerSpec"]
            container.pop("args", None)
            container["env"] = sorted((entry for entry in container["env"] if entry["name"] not in SOURCE_ENV),
                                      key=lambda entry: entry["name"])
        if old_fixed != new_fixed:
            raise ValueError("Template may only change the source bundle and its verified bootstrap")
        receipt_path = (self.root / receipt_path).resolve()
        if not receipt_path.is_relative_to(self.root):
            raise ValueError("Source receipt must be inside the session directory")
        receipt = json.loads(receipt_path.read_text())
        archive = (self.root / receipt["path"]).resolve()
        digest = receipt["sha256"]
        if (not archive.is_relative_to(self.root) or not isinstance(digest, str) or not SHA256.fullmatch(digest)
                or sha(archive) != digest or archive.stat().st_size != receipt["bytes"]
                or receipt["uri"] != f"{BUCKET}/bundles/{digest}/{archive.name}"):
            raise ValueError("Source bundle does not match its immutable local receipt")
        if after.get("HOLOSPEX_SOURCE_SHA256") != digest or after.get("HOLOSPEX_SOURCE_URI") != receipt["uri"]:
            raise ValueError("Template source identity does not match the verified local bundle")
        bootstrap_path = "holospex/ml/cloud/vertex_bootstrap.py"
        members = [item for item in receipt["members"] if item["path"] == bootstrap_path]
        with tarfile.open(archive, "r:gz") as packed:
            matches = [member for member in packed.getmembers() if member.name == bootstrap_path]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 1024 * 1024:
                raise ValueError("Source archive must contain one regular bootstrap file")
            with packed.extractfile(matches[0]) as stream:
                bootstrap = stream.read()
        if (len(members) != 1 or members[0]["sha256"] != hashlib.sha256(bootstrap).hexdigest()
                or members[0]["bytes"] != len(bootstrap)
                or template["workerPoolSpecs"][0]["containerSpec"].get("args") != [bootstrap.decode("utf-8")]):
            raise ValueError("Template bootstrap must exactly match the source bundle member")
        return receipt

    def apply_next_plan(self):
        """Atomically consume a valid local proposal once, including on restart."""
        path = self.root / "next-plan.json"
        if not path.exists():
            return False
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if self.state.get("rejected_proposal_sha256") == digest:
            return False
        try:
            proposal = json.loads(content)
            if not isinstance(proposal, dict) or set(proposal) - {"proposal_id", "queue", "template", "source_bundle"}:
                raise ValueError("Proposal contains unsupported fields; deadline and budgets cannot change")
            identity = proposal.get("proposal_id")
            if not isinstance(identity, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", identity):
                raise ValueError("Proposal needs a short unique proposal_id")
            applied = self.state.get("applied_proposals", {})
            if identity in applied:
                if applied[identity]["sha256"] != digest:
                    raise ValueError("Consumed proposal_id cannot be reused with different contents")
                return False
            if not ({"queue", "template"} & set(proposal)) or ("source_bundle" in proposal and "template" not in proposal):
                raise ValueError("Proposal needs a queue or a template; source_bundle requires template")
            if "queue" in proposal:
                if not isinstance(proposal["queue"], list) or len(proposal["queue"]) > MAX_LAUNCHES:
                    raise ValueError("Proposal queue must be a list of at most 16 recipes")
                for experiment in proposal["queue"]:
                    validate_recipe(experiment)
                members = [(item["replicate_best_full"], item["seed"]) for item in proposal["queue"] if "replicate_best_full" in item]
                if len(members) != len(set(members)):
                    raise ValueError("Replication queue cannot repeat a group/seed member")
            source = None
            if "template" in proposal:
                source = self.validate_template_proposal(proposal["template"], proposal.get("source_bundle", "source-bundle.json"))
            # All validation precedes mutation. The atomic state write records
            # the consumed ID together with the new plan, never either alone.
            previous = self.state
            updated = copy.deepcopy(previous)
            for key in ("queue", "template"):
                if key in proposal:
                    updated[key] = copy.deepcopy(proposal[key])
            if source is not None:
                updated["source"] = source
            updated.setdefault("applied_proposals", {})[identity] = {"sha256": digest, "applied_at": now().isoformat()}
            updated.pop("proposal_error", None)
            updated.pop("rejected_proposal_sha256", None)
            self.state = updated
            try:
                write_json(self.path, self.state)
            except BaseException:
                self.state = previous
                raise
        except Exception as error:
            self.state.update(proposal_error=str(error), rejected_proposal_sha256=digest)
            self.save()
            self.event("proposal_rejected", sha256=digest, error=str(error))
            return False
        self.event("proposal_applied", proposal_id=identity, sha256=digest,
                   source_sha256=self.state.get("source", {}).get("sha256"), queue_length=len(self.state["queue"]))
        self.save()
        return True

    def resolve_replication(self, marker):
        """Freeze one source recipe for a named pair before its first submission."""
        validate_recipe(marker)
        identity = marker["replicate_best_full"]
        groups = self.state.get("replication_groups", {})
        if not isinstance(groups, dict):
            raise ValueError("Replication groups must be an object")
        if any(run.get("replication", {}).get("group") == identity and run["recipe"]["seed"] == marker["seed"]
               for run in self.state["runs"]):
            raise ValueError("Replication group/seed already has a persisted launch intent")
        group = groups.get(identity)
        if identity not in groups:
            best = best_full_run(self.state["runs"])
            if best is None:
                raise ValueError("Replication marker requires a fully completed verified owned run")
            frozen = copy.deepcopy(best["recipe"])
            # Make existing optional defaults explicit in the resolved config.
            frozen.setdefault("auxiliary_loss_weight", 0.4)
            frozen.setdefault("lovasz_weight", 0.25)
            group = {"source_run_name": best["name"], "source_checkpoint_sha256": best["audit"]["checkpoint_sha256"],
                     "source_foreground_macro_iou": best["audit"]["foreground_macro_iou"],
                     "source_epochs": best["audit"]["epochs_completed"], "selected_at": now().isoformat(),
                     "selection": "best_verified_fully_completed_owned_native_validation_iou", "recipe": frozen,
                     "recipe_sha256": hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
        if (not isinstance(group, dict) or not isinstance(group.get("recipe"), dict)
                or "replicate_best_full" in group["recipe"]
                or not isinstance(group.get("source_run_name"), str) or not group["source_run_name"]
                or not isinstance(group.get("source_checkpoint_sha256"), str) or not SHA256.fullmatch(group["source_checkpoint_sha256"])
                or type(group.get("source_epochs")) is not int or group["source_epochs"] != group["recipe"].get("epochs")
                or group.get("recipe_sha256") != hashlib.sha256(json.dumps(group["recipe"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()):
            raise ValueError("Frozen replication group provenance or recipe is invalid")
        validate_recipe(group["recipe"])
        resolved = {**copy.deepcopy(group["recipe"]), **{key: marker[key] for key in ("label", "reason", "seed")}}
        resolved["reason"] += f" Frozen group {identity}; repeat full recipe from {group['source_run_name']} with seed {marker['seed']}."
        late = self.remaining() < 1500
        if late:
            resolved["reason"] += f" Late wall-clock budget: retain all {group['source_epochs']} requested epochs, but count as a full replication only if all epochs complete."
        validate_recipe(resolved)
        self.state.setdefault("replication_groups", {})[identity] = copy.deepcopy(group)
        metadata = {"group": identity, "source_run_name": group["source_run_name"],
                    "source_checkpoint_sha256": group["source_checkpoint_sha256"], "source_recipe_sha256": group["recipe_sha256"],
                    "source_epochs": group["source_epochs"], "full_recipe_preserved": True, "late_budget_warning": late}
        return resolved, metadata

    def launch(self, experiment):
        if (self.state["status"] not in {"ready", "running"}
                or self.remaining() < MIN_LAUNCH_SECONDS
                or len(self.state["runs"]) >= self.state["max_runs"]
                or sum(run["state"] not in TERMINAL for run in self.state["runs"]) >= self.state["max_parallel"]
                or any(run.get("inspection_error") for run in self.state["runs"])):
            raise RuntimeError("Controller launch budget, deadline, or unresolved-state guard prevents submission")
        replication = None
        if "replicate_best_full" in experiment:
            experiment, replication = self.resolve_replication(experiment)
        auxiliary_weight = _auxiliary_loss_weight(experiment)
        validate_initialization(experiment)
        number = len(self.state["runs"]) + 1
        name = f"{self.state['prefix']}-{number:03d}-{experiment['label']}"
        config = copy.deepcopy(self.state["template"])
        container = config["workerPoolSpecs"][0]["containerSpec"]
        env = {entry["name"]: entry["value"] for entry in container["env"]}
        for key in list(env):
            if key.startswith(("HOLOSPEX_INITIAL_", "HOLOSPEX_BACKBONE_CHECKPOINT_")):
                del env[key]
        env.update(HOLOSPEX_RUN_NAME=name, HOLOSPEX_OUTPUT_URI=f"{BUCKET}/runs/{name}")
        for key in ("architecture", "epochs", "width", "height", "lr", "lr_schedule", "warmup_epochs",
                    "weight_decay", "backbone_lr_multiplier", "augmentation", "sampling", "loss", "dice_weight", "seed"):
            env["HOLOSPEX_" + key.upper()] = str(experiment[key])
        # Optional overlap weight is explicit so an earlier template cannot
        # leak its loss weight into a later experiment.
        env["HOLOSPEX_LOVASZ_WEIGHT"] = str(experiment.get("lovasz_weight", 0.25))
        env["HOLOSPEX_AUXILIARY_LOSS_WEIGHT"] = str(auxiliary_weight)
        env["HOLOSPEX_DEADLINE_UTC"] = self.state["deadline"]
        if experiment.get("initial_checkpoint"):
            checkpoint = experiment["initial_checkpoint"]
            env.update(HOLOSPEX_INITIAL_CHECKPOINT_URI=checkpoint["uri"],
                       HOLOSPEX_INITIAL_CHECKPOINT_SHA256=checkpoint["sha256"])
        if experiment.get("backbone_checkpoint"):
            checkpoint = experiment["backbone_checkpoint"]
            env.update(HOLOSPEX_BACKBONE_CHECKPOINT_URI=checkpoint["uri"],
                       HOLOSPEX_BACKBONE_CHECKPOINT_SHA256=checkpoint["sha256"])
        env["HOLOSPEX_MAX_DURATION_SECONDS"] = str(max(60, min(3300, self.remaining() - 360)))
        config["scheduling"].update(timeout=f"{max(60, int(self.remaining()) - 10)}s", disableRetries=True)
        container["env"] = [{"name": key, "value": value} for key, value in sorted(env.items())]
        config_path = self.root / "jobs" / (name + ".json")
        write_json(config_path, config)
        run = {"name": name, "recipe": experiment, "config": str(config_path), "config_sha256": sha(config_path),
               "output_uri": env["HOLOSPEX_OUTPUT_URI"], "state": "SUBMIT_INTENT", "submitted_at": now().isoformat()}
        if replication is not None:
            run["replication"] = {**replication, "max_duration_seconds": float(env["HOLOSPEX_MAX_DURATION_SECONDS"])}
        run["bundle_references"] = {kind: {field: env.get(f"HOLOSPEX_{kind.upper()}_{field.upper()}")
                                                   for field in ("uri", "sha256")}
                                    for kind in ("source", "data", "prepared")}
        self.state["runs"].append(run)
        self.save()  # A restart must reconcile this intent; never repeat create.
        response = json.loads(command(["ai", "custom-jobs", "create", f"--project={PROJECT}", f"--region={REGION}",
                                       f"--display-name=holospex-{name}", f"--config={config_path}", "--format=json"]))
        run.update(job_name=response["name"], state=response["state"])
        write_json(self.root / "jobs" / (name + ".submitted.json"), response)
        self.event("submitted", name=name, job_name=run["job_name"], reason=experiment["reason"])
        self.save()

    def reconcile(self, run, *, audit=True):
        # Normal inspection must return by the absolute deadline too. During
        # shutdown, use a short independent timeout to confirm/cancel owned jobs.
        def inspection_timeout():
            return min(120, max(0.1, self.remaining())) if audit else 30

        if "job_name" not in run:
            response = json.loads(command(["ai", "custom-jobs", "list", f"--project={PROJECT}", f"--region={REGION}",
                                           f"--filter=displayName=holospex-{run['name']}", "--format=json"], timeout=inspection_timeout()))
            if len(response) != 1:
                raise RuntimeError(f"Ambiguous submit for {run['name']}; inspect intent without resubmission")
            run["job_name"] = response[0]["name"]
        response = json.loads(command(["ai", "custom-jobs", "describe", run["job_name"],
                                       f"--project={PROJECT}", f"--region={REGION}", "--format=json"], timeout=inspection_timeout()))
        previous = run["state"]
        run["state"] = response["state"]
        write_json(self.root / "jobs" / (run["name"] + ".status.json"), response)
        if run["state"] != previous:
            self.event("state", name=run["name"], state=run["state"])
        if audit and run["state"] == "JOB_STATE_SUCCEEDED" and "audit" not in run:
            budget = min(COLLECTION_TIMEOUT_SECONDS, self.remaining() - COLLECTION_DEADLINE_RESERVE_SECONDS)
            if budget > 0:
                self.collect_successful_run(run, timeout_seconds=budget)
        if run["state"] in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
            run["error"] = response.get("error")

    def collect_successful_run(self, run, *, timeout_seconds):
        """Bound both discovery and all downloads before validating a success."""
        deadline = time.monotonic() + timeout_seconds

        def remaining():
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError("Successful-run collection exhausted its total time budget")
            return value

        uris = [line for line in command(["storage", "ls", run["output_uri"] + "/attempts/*/status/completed.json"],
                                        timeout=min(120, remaining())).splitlines() if line.startswith("gs://")]
        if len(uris) != 1:
            raise RuntimeError("Exactly one completed attempt is required")
        destination = self.root / "results" / run["name"]
        if not destination.exists():
            collect(uris[0], destination, timeout_seconds=remaining())
        remaining()
        verified = summarize_run(destination)
        completion = json.loads((destination / "cloud-completion.json").read_text())
        checkpoint = completion["artifacts"]["train/best.pt"]
        if not checkpoint.get("uri", "").startswith("gs://") or checkpoint.get("sha256") != verified.get("checkpoint_sha256"):
            raise ValueError("Collected checkpoint reference does not match the verified artifact")
        write_json(destination / "independent-audit.json", verified)
        run.update(audit=verified, checkpoint_reference=checkpoint, result_dir=str(destination))
        self.event("evaluated", name=run["name"], **verified)

    def audit_after_stopping(self):
        """After compute is terminal, preserve late successes without waiting forever."""
        if any(run["state"] not in TERMINAL for run in self.state["runs"]):
            raise RuntimeError("Final collection requires all owned compute to be terminal")
        deadline = time.monotonic() + FINAL_COLLECTION_TIMEOUT_SECONDS
        for run in self.state["runs"]:
            if run["state"] != "JOB_STATE_SUCCEEDED" or "audit" in run:
                continue
            try:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    raise TimeoutError("Final successful-run collection budget exhausted; collect manually")
                self.collect_successful_run(run, timeout_seconds=budget)
                run.pop("inspection_error", None)
            except Exception as error:
                run["inspection_error"] = str(error)
                self.event("final_audit_error", name=run["name"], error=str(error))
            self.save()

    def stop_owned_jobs(self):
        for run in self.state["runs"]:
            if run["state"] not in TERMINAL:
                try:
                    # A create timeout may hide a running job. Resolve its
                    # persisted intent before deciding that all compute ended.
                    self.reconcile(run, audit=False)
                    if run["state"] not in TERMINAL and not run.get("cancellation_requested_at"):
                        command(["ai", "custom-jobs", "cancel", run["job_name"], f"--project={PROJECT}", f"--region={REGION}", "--quiet"], timeout=30)
                        run["cancellation_requested_at"] = now().isoformat()
                        self.event("cancel_requested", name=run["name"])
                    run.pop("inspection_error", None)
                except Exception as error:
                    run["inspection_error"] = str(error)
                    self.event("cancel_error", name=run["name"], error=str(error))
        return all(run["state"] in TERMINAL for run in self.state["runs"])

    def run(self):
        # Concurrent restarts must not spend against different snapshots of the
        # same budget. The OS releases this lock if the controller crashes.
        with (self.root / "controller.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another controller already owns this state file") from error
            self.__init__(self.path)  # Refresh only after acquiring ownership.
            if self.state["status"] not in RESUMABLE:
                raise ValueError(f"Refusing to restart inactive controller state {self.state['status']!r}")
            self.state.update(controller_pid=os.getpid(), controller_started_at=now().isoformat(), controller_status="running")
            self.state.pop("controller_stopped_at", None)
            self.save()
            outcome = "stopped"
            try:
                return self._run()
            except (KeyboardInterrupt, InterruptedError):
                outcome = "interrupted"
                raise
            except BaseException:
                outcome = "error"
                raise
            finally:
                self.state.update(controller_pid=None, controller_stopped_at=now().isoformat(), controller_status=outcome)
                self.save()

    def _run(self):
        if self.state["status"] not in RESUMABLE:
            raise ValueError(f"Refusing to restart inactive controller state {self.state['status']!r}")
        if self.state["status"] != "stopping":
            self.state["status"] = "running"
        self.save()
        while True:
            deadline = self.remaining() <= 0
            target = target_reached(self.state["runs"])
            exhausted = self.batch_exhausted()
            if deadline or target or exhausted or self.state["status"] == "stopping":
                self.state.setdefault("stop_reason", "target_reached" if target else
                                      "experiment_batch_complete" if exhausted and not deadline else "deadline_reached")
                self.state["status"] = "stopping"
                self.save()
                if self.stop_owned_jobs():
                    self.audit_after_stopping()
                    self.state["status"] = self.state["stop_reason"]
                    self.save()
                    self.event("finished", status=self.state["status"])
                    return
                self.save()
                time.sleep(min(60, max(1, self.state.get("poll_seconds", 45))))
                continue
            self.apply_next_plan()
            for run in self.state["runs"]:
                if run["state"] not in TERMINAL or (run["state"] == "JOB_STATE_SUCCEEDED" and "audit" not in run):
                    try:
                        self.reconcile(run)
                        run.pop("inspection_error", None)
                    except Exception as error:
                        run["inspection_error"] = str(error)
                        self.event("inspection_error", name=run["name"], error=str(error))
                    self.save()
                    if self.remaining() <= 0 or target_reached(self.state["runs"]):
                        break
            if self.remaining() <= 0 or target_reached(self.state["runs"]) or self.batch_exhausted():
                continue
            active = [r for r in self.state["runs"] if r["state"] not in TERMINAL]
            # Reconcile/audit errors block new spend, but existing jobs keep running.
            unresolved = any(r.get("inspection_error") for r in self.state["runs"])
            while (len(active) < self.state["max_parallel"] and not unresolved
                   and len(self.state["runs"]) < self.state["max_runs"]
                   and self.remaining() >= MIN_LAUNCH_SECONDS):
                experiment = choose_next(self.state, self.remaining())
                if experiment is None:
                    break
                try:
                    self.launch(experiment)
                except Exception as error:
                    self.event("launch_error", error=str(error))
                    break
                active = [r for r in self.state["runs"] if r["state"] not in TERMINAL]
            self.save()
            time.sleep(min(60, max(1, self.state.get("poll_seconds", 45)), max(0.1, self.remaining())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    Controller(args.state).run()


if __name__ == "__main__":
    main()
