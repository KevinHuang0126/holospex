"""Run one fresh CUDA experiment and preserve artifacts in a private GCS bucket.

The runtime bootstrap owns installing dependencies and transferring source/data.
This entry point uses ADC from the Vertex service account; it never accepts keys.
Spot retries start separate attempts. Checkpoints are useful artifacts, not exact
resumption state: the training CLI does not save optimizer or RNG state.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from urllib.parse import urlsplit


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_output_uri(uri):
    parsed = urlsplit(uri)
    parts = parsed.path.strip("/").split("/")
    if (parsed.scheme != "gs" or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", parsed.netloc)
            or parsed.query or parsed.fragment or not parsed.path.startswith("/")
            or any(not part or part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in parts)):
        raise ValueError("output-uri must be gs://bucket/nonempty/path with simple path components.")
    return parsed.netloc, "/".join(parts)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def stable_copy(source, destination):
    """Copy a stable inode; reject both in-place writes and atomic replacement.

    Training atomically replaces checkpoints/history. Opening the old inode is
    safe, but a replacement during this snapshot still causes a conservative
    retry so one snapshot does not silently mix epochs.
    """
    before = signature(source)
    shutil.copyfile(source, destination)
    if before != signature(source):
        destination.unlink(missing_ok=True)
        return None
    return before


def read_strict_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=invalid_constant)


def contained_file(root, relative):
    """Resolve ordinary package files without following symlinks or host paths."""
    if not isinstance(relative, str):
        raise ValueError("Package paths must be strings")
    parts = PurePosixPath(relative)
    if (not relative or parts.is_absolute() or ".." in parts.parts
            or parts.as_posix() != relative or "\\" in relative):
        raise ValueError(f"Unsafe package path: {relative}")
    target = root
    for part in parts.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f"Symlink in package path: {relative}")
    if not target.is_file() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Missing or escaping package file: {relative}")
    return target.resolve()


def verify_file(path, expected, size=None):
    if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or (size is not None and (type(size) is not int or size < 0))):
        raise ValueError("Package file requires a SHA-256 and nonnegative integer size")
    if (size is not None and path.stat().st_size != size) or sha256(path) != expected:
        raise ValueError(f"Package file integrity mismatch: {path.name}")


def prepare_packaged_manifest(package_path, data_root, output):
    """Validate immutable local inputs, then rebase file references for this host.

    The package preserves the exact original manifests/preparation artifacts.
    No directory search or basename fallback may silently select another mask.
    Its complete file inventory is bound by the bootstrap's archive SHA-256.
    """
    import copy
    import numpy as np
    from PIL import Image

    package_path, data_root = package_path.resolve(), data_root.resolve()
    package_root = package_path.parent
    package = read_strict_json(package_path)
    if (package.get("formatVersion") != "1.0.0"
            or package.get("artifactType") != "holospex_prepared_cloud_package"):
        raise ValueError("Unsupported prepared cloud package")
    manifest_path = contained_file(package_root, "manifest.json")
    base_path = contained_file(package_root, "base-manifest.json")
    verify_file(manifest_path, package.get("manifestSha256"))
    verify_file(base_path, package.get("baseManifestSha256"))
    manifest, base = read_strict_json(manifest_path), read_strict_json(base_path)
    files, destinations = {}, set()
    for record in package.get("files", []):
        source, kind = record.get("sourcePath"), record.get("root")
        if not isinstance(source, str) or not Path(source).is_absolute() or source in files:
            raise ValueError("Package source paths must be unique absolute paths")
        if kind not in {"data", "prepared"}:
            raise ValueError("Package file root must be data or prepared")
        target = contained_file(data_root if kind == "data" else package_root, record.get("path"))
        if target in destinations or "bytes" not in record:
            raise ValueError("Duplicate package destination or missing byte count")
        verify_file(target, record.get("sha256"), record["bytes"])
        files[source] = {**record, "target": target}
        destinations.add(target)

    provenance = manifest.get("reviewedPartialProvenance", {})
    summary_source = provenance.get("preparationPath")
    if summary_source not in files:
        raise ValueError("Prepared manifest requires its verified preparation summary")
    summary = read_strict_json(files[summary_source]["target"])
    if summary.get("artifactType") != "reviewed_partial_training_preparation":
        raise ValueError("Invalid preparation summary")
    if (summary.get("combinedManifestSha256") != package["manifestSha256"]
            or summary.get("baseManifestSha256") != package["baseManifestSha256"]):
        raise ValueError("Preparation summary manifest hashes disagree")
    for key in ("bundleSha256", "reviewSha256", "baseManifestSha256", "resolutionRecordSha256"):
        if not summary.get(key) or provenance.get(key) != summary[key]:
            raise ValueError(f"Preparation provenance mismatch: {key}")
    if (provenance.get("overlapPolicy") != "ignore_conflicts"
            or provenance.get("unknownPixelPolicy") != "ignore"
            or provenance.get("reviewer", {}).get("reviewScope") != "anatomy"):
        raise ValueError("Prepared labels require anatomy review and explicit ignore policies")
    original_root = Path(summary_source).parent
    for record in summary.get("artifactInventory", []):
        # Validate the relative path even though source references are not read.
        target = contained_file(package_root, record.get("path"))
        source = str(original_root / record["path"])
        if source not in files or files[source]["target"] != target:
            raise ValueError("Preparation artifact missing from package inventory")
        verify_file(target, record.get("sha256"))
    base_records = {}
    for record in summary.get("baseFiles", []):
        for prefix in ("image", "mask"):
            source = record[f"{prefix}Path"]
            if (source in base_records or source not in files
                    or files[source]["sha256"] != record[f"{prefix}Sha256"]):
                raise ValueError("Base data differs from the reviewed preparation")
            base_records[source] = record[f"{prefix}Sha256"]

    for key in ("schemaVersion", "classes", "ignoreIndex", "ignoreSourceIds"):
        if manifest.get(key) != base.get(key):
            raise ValueError(f"Base/prepared manifest policy mismatch: {key}")
    if manifest.get("dataset") != base.get("dataset", "") + "+reviewed-partial":
        raise ValueError("Prepared dataset must identify the original plus reviewed partial data")
    classes = manifest.get("classes", [])
    known = {item["sourceId"] for item in classes}
    if (len(classes) < 2 or [item["index"] for item in classes] != list(range(len(classes)))
            or len(known) != len(classes) or classes[0].get("structureId") != "background"
            or classes[0].get("sourceId") != 0 or manifest.get("ignoreIndex") != 255
            or manifest.get("ignoreSourceIds") != [255] or 255 in known):
        raise ValueError("Invalid prepared source/model class mapping or ignore policy")
    originals, samples = base.get("samples", []), manifest.get("samples", [])
    if not originals or samples[:len(originals)] != originals or len(samples) <= len(originals):
        raise ValueError("Prepared samples must preserve the complete original ordered prefix and add training images")
    base_cases = {str(sample["videoId"]) for sample in originals}
    cases, seen, counts = {}, set(), {"train": 0, "val": 0, "test": 0}
    rebased = copy.deepcopy(manifest)
    for index, (sample, destination) in enumerate(zip(samples, rebased["samples"])):
        split, case = sample.get("split"), str(sample.get("videoId"))
        key = (case, sample.get("frameNumber"))
        if split not in counts or sample.get("videoId") is None or key in seen:
            raise ValueError("Invalid or duplicate sample identity/split")
        if case in cases and cases[case] != split:
            raise ValueError("Surgical case leakage across data splits")
        cases[case], counts[split] = split, counts[split] + 1
        seen.add(key)
        added = index >= len(originals)
        if added and (split != "train" or case in base_cases
                      or sample.get("annotationSource") != "reviewed_partial_anatomy"):
            raise ValueError("Added samples must be reviewed partial masks from new training cases")
        for field in ("imagePath", "maskPath", "reviewProvenancePath"):
            if field not in sample and field == "reviewProvenancePath" and not added:
                continue
            source = sample.get(field)
            if source not in files:
                raise ValueError(f"Sample {field} missing from verified inventory")
            if not added and field in {"imagePath", "maskPath"} and source not in base_records:
                raise ValueError("Original sample missing from preparation base-file record")
            destination[field] = str(files[source]["target"])
        with Image.open(destination["imagePath"]) as image, Image.open(destination["maskPath"]) as mask_image:
            mask = np.asarray(mask_image)
            if mask_image.format != "PNG" or mask.ndim != 2 or image.size != mask_image.size:
                raise ValueError("Image/mask shape or semantic PNG format mismatch")
            labels = set(np.unique(mask).tolist())
            if labels - known - {255}:
                raise ValueError("Unknown semantic source IDs in a training mask")
            if added and (0 in labels or not labels - {255}):
                raise ValueError("Partial targets cannot contain fabricated background or only ignored pixels")
    if counts != manifest.get("report", {}).get("counts") or counts != summary.get("summary", {}).get("combinedCounts"):
        raise ValueError("Prepared sample counts disagree with reports")
    if (len(samples) - len(originals) != summary.get("summary", {}).get("addedTrainImageCount")
            or len(originals) != summary.get("summary", {}).get("baseSampleCount")
            or len(summary.get("baseFiles", [])) != len(originals)):
        raise ValueError("Preparation source or added sample counts disagree")
    rebased["root"] = str(data_root)
    rebased["reviewedPartialProvenance"]["preparationPath"] = str(files[summary_source]["target"])
    verification = {"artifactType": "prepared_cloud_manifest_verification", "verified": True,
                    "packageSha256": sha256(package_path), "sourceManifestSha256": sha256(manifest_path),
                    "baseManifestSha256": sha256(base_path), "verifiedFileCount": len(files),
                    "counts": counts, "addedTrainImageCount": len(samples) - len(originals),
                    "originalSamplesPreserved": True, "heldOutSamplesPreserved": True,
                    "rebasingScope": "root, sample image/mask/provenance paths, preparationPath"}
    # Validation finishes before any run manifest is published.
    (output / "manifest.json").write_bytes(json_bytes(rebased))
    (output / "prepared-input-verification.json").write_bytes(json_bytes(verification))
    shutil.copyfile(package_path, output / "prepared-package.json")
    shutil.copyfile(manifest_path, output / "manifest-source.json")
    shutil.copyfile(base_path, output / "base-manifest-source.json")
    return verification


def audit_training_inputs(output):
    """Bind both experiment arms to path-independent, ordered data identities."""
    manifest_path = output / "manifest.json"
    manifest = read_strict_json(manifest_path)
    splits = {split: [] for split in ("train", "val", "test")}
    for sample in manifest["samples"]:
        identity = {key: value for key, value in sample.items()
                    if key not in {"imagePath", "maskPath", "reviewProvenancePath"}}
        identity["imageSha256"] = sha256(Path(sample["imagePath"]))
        identity["maskSha256"] = sha256(Path(sample["maskPath"]))
        splits[sample["split"]].append(identity)
    report = {"artifactType": "training_input_audit", "manifestSha256": sha256(manifest_path),
              "splitCounts": {split: len(rows) for split, rows in splits.items()},
              "splitCaseIds": {split: sorted({str(row["videoId"]) for row in rows}) for split, rows in splits.items()},
              "splitContentSha256": {split: hashlib.sha256(json_bytes(rows)).hexdigest() for split, rows in splits.items()},
              "fingerprintScope": "Ordered sample metadata excluding host file paths, plus exact image and mask SHA-256",
              "classes": manifest["classes"], "ignoreIndex": manifest.get("ignoreIndex"),
              "ignoreSourceIds": manifest.get("ignoreSourceIds"), "samples": splits}
    (output / "training-input-audit.json").write_bytes(json_bytes(report))
    return report


class ArtifactStore:
    def __init__(self, bucket, prefix):
        self.bucket, self.prefix = bucket, prefix
        self.known = {}

    def _upload(self, name, digest, size, action):
        key = f"{self.prefix}/{name}"
        if key in self.known and self.known[key] != digest:
            raise RuntimeError(f"Immutable artifact collision: {key}")
        if key not in self.known:
            blob = self.bucket.blob(key)
            blob.metadata = {"sha256": digest}
            try:
                action(blob)
            except Exception as error:
                # A retried request may have succeeded before its response was
                # lost. Never replace an object: verify its existing identity.
                if getattr(error, "code", None) != 412:
                    raise
                existing = self.bucket.get_blob(key)
                if existing is None or (existing.metadata or {}).get("sha256") != digest or existing.size != size:
                    raise RuntimeError(f"Immutable artifact collision: {key}") from error
            self.known[key] = digest
        return {"uri": f"gs://{self.bucket.name}/{key}", "sha256": digest, "bytes": size}

    def file(self, path, logical_name):
        digest, size = sha256(path), path.stat().st_size
        name = f"objects/{digest}/{Path(logical_name).name}"
        return self._upload(name, digest, size, lambda blob: blob.upload_from_filename(str(path), if_generation_match=0))

    def record(self, name, value):
        payload = json_bytes(value)
        digest = hashlib.sha256(payload).hexdigest()
        return self._upload(name, digest, len(payload), lambda blob: blob.upload_from_string(payload, content_type="application/json", if_generation_match=0))


def read_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


EPOCH_FILES = ("train/config.json", "train/history.json", "train/last.pt", "train/best.pt", "train/metrics-validation.json")


def verify_epoch(files, checkpoint_reader=read_checkpoint):
    config = json.loads(files["train/config.json"].read_text())
    history = json.loads(files["train/history.json"].read_text())
    if not history or [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        return None
    best_row = max(history, key=lambda row: row["validation"]["foreground_macro_iou"])
    last = checkpoint_reader(files["train/last.pt"])
    best = checkpoint_reader(files["train/best.pt"])
    for checkpoint, row in ((last, history[-1]), (best, best_row)):
        if (checkpoint["epochs_trained"] != row["epoch"] or checkpoint["training"] != config
                or checkpoint["validation"] != row["validation"]):
            return None
    if json.loads(files["train/metrics-validation.json"].read_text()) != best_row["validation"]:
        return None
    return {"epoch": len(history), "best_epoch": best_row["epoch"], "last_model_version": last["model_version"],
            "best_model_version": best["model_version"], "run_id": last["run_id"]}


class Synchronizer:
    def __init__(self, root, store, checkpoint_reader=read_checkpoint):
        self.root, self.store, self.checkpoint_reader = root, store, checkpoint_reader
        self.latest = {}
        self.epoch_record = None
        self.last_epoch = 0

    def sync(self, final=False):
        # This method is called by the main thread only. The training subprocess
        # writes concurrently, using atomic replacement for epoch artifacts.
        with tempfile.TemporaryDirectory(prefix="holospex-sync-") as temporary:
            stage = Path(temporary)
            files, signatures = {}, {}
            if all((self.root / name).is_file() for name in EPOCH_FILES):
                for index, name in enumerate(EPOCH_FILES):
                    target = stage / str(index)
                    observed = stable_copy(self.root / name, target)
                    if observed is None:
                        break
                    signatures[name], files[name] = observed, target
                if len(files) == len(EPOCH_FILES) and all(signature(self.root / name) == observed for name, observed in signatures.items()):
                    epoch = verify_epoch(files, self.checkpoint_reader)
                    if epoch and epoch["epoch"] > self.last_epoch:
                        artifacts = {name: self.store.file(path, name) for name, path in files.items()}
                        self.epoch_record = self.store.record(f"epochs/{epoch['epoch']:04d}.json", {
                            **epoch, "artifacts": artifacts, "exact_resume_supported": False,
                        })
                        self.latest.update(artifacts)
                        self.last_epoch = epoch["epoch"]
                        print(f"Preserved coherent completed epoch {self.last_epoch}: {self.epoch_record['uri']}", flush=True)
            if final and self.last_epoch == 0:
                raise RuntimeError("No coherent completed epoch is available for final upload.")
            # Logs can change between polls. A skipped copy is retried next time;
            # stdout is also captured continuously by Vertex Cloud Logging.
            extras = [path for path in self.root.rglob("*") if path.is_file()
                      and str(path.relative_to(self.root)) not in EPOCH_FILES and path.suffix != ".tmp"]
            for index, source in enumerate(extras):
                target = stage / f"extra-{index}"
                if stable_copy(source, target) is not None:
                    logical = source.relative_to(self.root).as_posix()
                    self.latest[logical] = self.store.file(target, logical)


def build_commands(args, output):
    python = [sys.executable, "-u", "-m", "holospex_ml"]
    manifest = str(output / "manifest.json")
    train = python + ["train", "--manifest", manifest, "--output-dir", str(output / "train"),
                      "--device", "cuda", "--epochs", str(args.epochs), "--batch-size", "2", "--seed", str(args.seed),
                      "--width", str(args.width), "--height", str(args.height), "--lr", str(getattr(args, "lr", 0.0003)), "--class-weighting", "balanced"]
    for option in ("architecture", "augmentation", "sampling", "loss"):
        if getattr(args, option, None) is not None:
            train.extend([f"--{option}", getattr(args, option)])
    if getattr(args, "dice_weight", None) is not None:
        train.extend(["--dice-weight", str(args.dice_weight)])
    if getattr(args, "lovasz_weight", None) is not None:
        train.extend(["--lovasz-weight", str(args.lovasz_weight)])
    if getattr(args, "auxiliary_loss_weight", None) is not None:
        train.extend(["--auxiliary-loss-weight", str(args.auxiliary_loss_weight)])
    for option in ("lr_schedule", "warmup_epochs", "weight_decay", "backbone_lr_multiplier", "max_duration_seconds", "initial_checkpoint", "backbone_checkpoint"):
        value = getattr(args, option, None)
        if value is not None:
            train.extend(["--" + option.replace("_", "-"), str(value)])
    commands = [
        ("requirements", [sys.executable, "-m", "pip", "freeze"], "requirements-runtime.txt"),
        ("environment", python + ["doctor"], "environment.json"),
        ("prepare", python + ["prepare-endoscapes", "--data-root", str(args.data_root.resolve()),
                              "--fps", "25", "--ignore-source-id", "255", "--exclude-frame", "153_32700",
                              "--output", manifest], "prepare.log"),
        ("train", train, "train.log"),
        ("evaluate", python + ["evaluate-original", "--manifest", manifest, "--checkpoint", str(output / "train/best.pt"),
                               "--split", "val", "--device", "cuda", "--output", str(output / "train/metrics-val-original.json")], "evaluate.log"),
    ]
    if getattr(args, "prepared_package", None) is not None:
        commands = [item for item in commands if item[0] != "prepare"]
    return commands


def run_command(command, log_path, synchronizer, interval):
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", buffering=1) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=environment)

        def relay():
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)

        reader = threading.Thread(target=relay, daemon=True)
        reader.start()
        try:
            while True:
                try:
                    code = process.wait(timeout=interval)
                    break
                except subprocess.TimeoutExpired:
                    # A storage outage must not quietly discard hours of work.
                    # The client retries transient requests; exhausted failures
                    # fail the job, whose next attempt starts in a new prefix.
                    synchronizer.sync()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            reader.join(timeout=15)
            process.stdout.close()
        if code:
            raise subprocess.CalledProcessError(code, command)
    synchronizer.sync()


def validate_optimization_args(args):
    """Validate locally before creating an attempt or contacting cloud storage."""
    for option in ("lr", "weight_decay", "backbone_lr_multiplier", "max_duration_seconds", "dice_weight", "lovasz_weight", "sync_seconds"):
        value = getattr(args, option, None)
        if value is None:
            continue
        allow_zero = option in {"weight_decay", "dice_weight", "lovasz_weight"}
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or (value == 0 and not allow_zero)):
            raise ValueError(f"{option} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    auxiliary_weight = getattr(args, "auxiliary_loss_weight", None)
    if auxiliary_weight is not None and (isinstance(auxiliary_weight, bool) or not isinstance(auxiliary_weight, (int, float)) or not math.isfinite(auxiliary_weight) or not 0 <= auxiliary_weight <= 10):
        raise ValueError("auxiliary_loss_weight must be finite and in [0, 10]")
    schedule = getattr(args, "lr_schedule", None)
    if schedule is not None and schedule not in {"none", "cosine"}:
        raise ValueError("lr_schedule must be none or cosine")
    warmup = getattr(args, "warmup_epochs", None)
    if warmup is not None and (type(warmup) is not int or not 0 <= warmup < args.epochs):
        raise ValueError("warmup_epochs must be an integer in [0, epochs)")
    initial = getattr(args, "initial_checkpoint", None)
    backbone = getattr(args, "backbone_checkpoint", None)
    if initial is not None and backbone is not None:
        raise ValueError("backbone-checkpoint and initial-checkpoint are mutually exclusive")
    if initial is not None and not initial.is_file():
        raise ValueError("initial-checkpoint must name an existing checkpoint file")
    if backbone is not None:
        if getattr(args, "architecture", None) != "deeplabv3_resnet50":
            raise ValueError("backbone-checkpoint requires architecture deeplabv3_resnet50")
        if not backbone.is_file():
            raise ValueError("backbone-checkpoint must name an existing checkpoint file")
    if getattr(args, "deadline_utc", None) is not None:
        parse_deadline(args.deadline_utc)


def parse_deadline(value, *, now=None):
    try:
        deadline = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("deadline-utc must be an aware ISO timestamp") from error
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("deadline-utc must include a timezone")
    deadline = deadline.astimezone(timezone.utc)
    current = now or datetime.fromisoformat(utc_now())
    if deadline <= current:
        raise ValueError("deadline-utc has expired; refusing new training")
    return deadline


def apply_deadline_budget(args, *, now=None):
    """Recompute the actual train cap after preparing and auditing input files."""
    if getattr(args, "deadline_utc", None) is None:
        return None
    current = now or datetime.fromisoformat(utc_now())
    deadline = parse_deadline(args.deadline_utc, now=current)
    remaining = (deadline - current).total_seconds()
    reserve = 120.0
    if remaining <= reserve:
        raise ValueError("At most 120 seconds remain before deadline; refusing to start another training phase")
    requested = getattr(args, "max_duration_seconds", None)
    effective = min(requested, remaining - reserve) if requested is not None else remaining - reserve
    args.max_duration_seconds = effective
    return {"deadline_utc": deadline.isoformat(), "observed_at": current.isoformat(),
            "remaining_seconds": remaining, "evaluation_and_upload_reserve_seconds": reserve,
            "requested_max_duration_seconds": requested, "effective_max_duration_seconds": effective,
            "training_stop_policy": "finish_current_epoch_and_save; epoch_boundary_may_exceed_effective_cap"}


def training_stop_reason(args, output, epochs_completed):
    """A requested cap alone does not establish a graceful duration stop.

    The CLI prints its final result only after returning successfully from
    training. Verify that result against coherent saved epoch artifacts and the
    immutable configuration before treating a short run as completed.
    """
    if not 1 <= epochs_completed <= args.epochs:
        raise RuntimeError("Training epoch artifacts are incomplete or exceed the requested count.")
    budget = getattr(args, "max_duration_seconds", None)
    if budget is None:
        if epochs_completed != args.epochs:
            raise RuntimeError("Training epoch artifacts are incomplete.")
        return {"stop_reason": "epochs_completed", "training_duration_seconds": None}
    log = (output / "train.log").read_text()
    starts = list(re.finditer(r"(?m)^\{", log))
    report = None
    for match in reversed(starts):
        try:
            candidate = json.loads(log[match.start():])
        except ValueError:
            continue
        if isinstance(candidate, dict):
            report = candidate
            break
    if (report is None or type(report.get("epochs_completed")) is not int
            or report["epochs_completed"] != epochs_completed):
        raise RuntimeError("Training artifacts are incomplete: final CLI report is missing or disagrees with saved epochs.")
    config = read_strict_json(output / "train/config.json")
    if config.get("epochs") != args.epochs or config.get("max_duration_seconds") != budget:
        raise RuntimeError("Training duration/epoch configuration differs from the requested experiment.")
    expected_reason = "max_duration_seconds" if epochs_completed < args.epochs else "epochs_completed"
    duration = report.get("duration_seconds")
    if (report.get("stop_reason") != expected_reason or isinstance(duration, bool)
            or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0
            or (expected_reason == "max_duration_seconds" and duration < budget)):
        raise RuntimeError("Training final report does not verify a completed epoch or graceful duration stop.")
    return {"stop_reason": expected_reason, "training_duration_seconds": duration}


def execute(args, bucket, *, run=run_command, checkpoint_reader=read_checkpoint):
    validate_optimization_args(args)
    # The effective worker budget must not mutate the submitting caller's args.
    args = copy.copy(args)
    _, prefix = parse_output_uri(args.output_uri)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    output = args.work_dir.resolve() / attempt
    output.mkdir(parents=True, exist_ok=False)
    store = ArtifactStore(bucket, f"{prefix}/attempts/{attempt}")
    synchronizer = Synchronizer(output, store, checkpoint_reader)
    identity = {"run_name": args.run_name, "attempt": attempt, "attempt_uri": f"gs://{bucket.name}/{store.prefix}",
                "started_at": utc_now(), "epochs_requested": args.epochs, "exact_resume_supported": False,
                "deadline_utc": getattr(args, "deadline_utc", None),
                "max_duration_seconds_requested": getattr(args, "max_duration_seconds", None),
                "retry_policy": "A new invocation starts fresh in a unique attempt directory."}
    commands = build_commands(args, output)
    (output / "commands.json").write_bytes(json_bytes(commands))
    print(json.dumps(identity), flush=True)
    try:
        store.record("status/running.json", {**identity, "state": "running"})
        if getattr(args, "prepared_package", None) is not None:
            verification = prepare_packaged_manifest(args.prepared_package, args.data_root, output)
            print(json.dumps(verification), flush=True)
            synchronizer.sync()
        for command_index, (phase, command, log) in enumerate(commands):
            if phase == "train":
                audit_training_inputs(output)
                synchronizer.sync()
                budget = apply_deadline_budget(args)
                if budget is not None:
                    (output / "deadline-budget.json").write_bytes(json_bytes(budget))
                    command = next(value for label, value, _ in build_commands(args, output) if label == "train")
                    commands[command_index] = (phase, command, log)
                    (output / "commands.json").write_bytes(json_bytes(commands))
            print(f"Starting phase: {phase}", flush=True)
            run(command, output / log, synchronizer, args.sync_seconds)
        synchronizer.sync(final=True)
        if "train/metrics-val-original.json" not in synchronizer.latest:
            raise RuntimeError("Training or original-resolution validation artifacts are incomplete.")
        termination = training_stop_reason(args, output, synchronizer.last_epoch)
        completed = {**identity, "state": "completed", "finished_at": utc_now(), "epochs_completed": synchronizer.last_epoch,
                     "budget_exhausted": termination["stop_reason"] == "max_duration_seconds", **termination,
                     "max_duration_seconds_effective": getattr(args, "max_duration_seconds", None),
                     "last_epoch_manifest": synchronizer.epoch_record, "artifacts": synchronizer.latest}
        store.record("status/completed.json", completed)
        print(json.dumps(completed, indent=2), flush=True)
        return completed
    except BaseException as error:
        try:
            synchronizer.sync()
        except Exception as upload_error:
            print(f"Could not sync artifacts after failure: {upload_error}", file=sys.stderr, flush=True)
        try:
            store.record("status/failed.json", {**identity, "state": "failed", "finished_at": utc_now(),
                         "error": f"{type(error).__name__}: {error}", "epochs_preserved": synchronizer.last_epoch,
                         "last_epoch_manifest": synchronizer.epoch_record, "artifacts": synchronizer.latest})
        except Exception as upload_error:
            print(f"Could not persist failure status: {upload_error}", file=sys.stderr, flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prepared-package", type=Path,
                        help="Verified prepared/package.json; skips original-only manifest regeneration")
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--architecture")
    parser.add_argument("--augmentation")
    parser.add_argument("--sampling")
    parser.add_argument("--loss", choices=["ce", "ce_generalized_dice", "ce_lovasz"])
    parser.add_argument("--dice-weight", type=float)
    parser.add_argument("--lovasz-weight", type=float)
    parser.add_argument("--auxiliary-loss-weight", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--lr-schedule", choices=["none", "cosine"])
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--backbone-lr-multiplier", type=float)
    parser.add_argument("--max-duration-seconds", type=float)
    parser.add_argument("--deadline-utc", help="Absolute aware ISO deadline, including a 120-second evaluation/upload reserve before training")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/holospex-runs"))
    parser.add_argument("--sync-seconds", type=float, default=15)
    args = parser.parse_args(argv)
    bucket_name, _ = parse_output_uri(args.output_uri)
    if args.epochs <= 0 or args.width <= 0 or args.height <= 0 or args.sync_seconds <= 0 or not args.data_root.is_dir():
        parser.error("Require existing data-root, positive epochs/dimensions, and positive sync-seconds.")
    if args.prepared_package is not None and not args.prepared_package.is_file():
        parser.error("prepared-package must name an existing package.json")
    try:
        validate_optimization_args(args)
    except ValueError as error:
        parser.error(str(error))
    from google.cloud import storage
    execute(args, storage.Client().bucket(bucket_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
