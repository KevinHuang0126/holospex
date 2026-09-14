"""Bootstrap a pinned Vertex GPU container from private, checksummed GCS bundles.

The submitting client embeds this file in ``python -u -c``. It needs only the
standard library until it explicitly installs the Storage SDK. Authentication
comes from the Vertex workload identity; never pass credentials in job config.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
from urllib.parse import urlsplit


def split_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("Expected a gs://bucket/object URI")
    if parsed.query or parsed.fragment or parsed.username or parsed.port:
        raise ValueError("GCS object URIs cannot contain credentials, query, or fragment")
    return parsed.netloc, parsed.path.lstrip("/")


def verify_sha256(path: Path, expected: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
        raise ValueError("Expected an explicit 64-character SHA-256")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    return actual


def safe_extract(archive_path: Path, destination: Path, top_level: str) -> None:
    """Extract only ordinary files/directories under one new, expected root."""
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / top_level).exists():
        raise ValueError(f"Refusing to overwrite existing {top_level} directory")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        seen: set[str] = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or not path.parts
                    or path.parts[0] != top_level):
                raise ValueError(f"Archive member outside {top_level}: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive member type: {member.name}")
            normalized = path.as_posix()
            if normalized in seen:
                raise ValueError(f"Duplicate archive member: {member.name}")
            seen.add(normalized)
        if not members:
            raise ValueError("Archive is empty")
        # Validate every member before writing any of them. In particular, do
        # not follow archive symlinks/hardlinks or preserve host ownership.
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable archive member: {member.name}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def entry_command(environment: dict[str, str], work: Path) -> list[str]:
    command = [
        sys.executable, "-u", str(work / "holospex/ml/cloud/vertex_entry.py"),
        "--data-root", str(work / "endoscapes"),
        "--output-uri", environment["HOLOSPEX_OUTPUT_URI"],
        "--run-name", environment["HOLOSPEX_RUN_NAME"],
        "--epochs", environment["HOLOSPEX_EPOCHS"],
    ]
    if prepared_bundle_enabled(environment):
        command.extend(["--prepared-package", str(work / "prepared/package.json")])
    for option in ("width", "height", "architecture", "augmentation", "sampling", "seed", "loss", "dice_weight", "lovasz_weight", "auxiliary_loss_weight",
                   "lr", "lr_schedule", "warmup_epochs", "weight_decay", "backbone_lr_multiplier", "max_duration_seconds"):
        value = environment.get("HOLOSPEX_" + option.upper())
        if value:
            command.extend(["--" + option.replace("_", "-"), value])
    if environment.get("HOLOSPEX_DEADLINE_UTC"):
        command.extend(["--deadline-utc", environment["HOLOSPEX_DEADLINE_UTC"]])
    if initial_checkpoint_enabled(environment):
        command.extend(["--initial-checkpoint", str(work / "initial.pt")])
    if backbone_checkpoint_enabled(environment):
        command.extend(["--backbone-checkpoint", str(work / "backbone.pt")])
    return command


def prepared_bundle_enabled(environment: dict[str, str]) -> bool:
    uri, digest = environment.get("HOLOSPEX_PREPARED_URI"), environment.get("HOLOSPEX_PREPARED_SHA256")
    if bool(uri) != bool(digest):
        raise ValueError("Set both HOLOSPEX_PREPARED_URI and HOLOSPEX_PREPARED_SHA256")
    if uri:
        split_gcs_uri(uri)
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError("Invalid HOLOSPEX_PREPARED_SHA256")
    return bool(uri)


def initial_checkpoint_enabled(environment: dict[str, str]) -> bool:
    uri = environment.get("HOLOSPEX_INITIAL_CHECKPOINT_URI")
    digest = environment.get("HOLOSPEX_INITIAL_CHECKPOINT_SHA256")
    if bool(uri) != bool(digest):
        raise ValueError("Initial checkpoint requires both URI and SHA-256")
    if uri:
        split_gcs_uri(uri)
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError("Invalid HOLOSPEX_INITIAL_CHECKPOINT_SHA256")
    return bool(uri)


def backbone_checkpoint_enabled(environment: dict[str, str]) -> bool:
    uri = environment.get("HOLOSPEX_BACKBONE_CHECKPOINT_URI")
    digest = environment.get("HOLOSPEX_BACKBONE_CHECKPOINT_SHA256")
    if bool(uri) != bool(digest):
        raise ValueError("Backbone checkpoint requires both URI and SHA-256")
    if uri:
        split_gcs_uri(uri)
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError("Invalid HOLOSPEX_BACKBONE_CHECKPOINT_SHA256")
        if environment.get("HOLOSPEX_ARCHITECTURE") != "deeplabv3_resnet50":
            raise ValueError("Backbone checkpoint requires HOLOSPEX_ARCHITECTURE=deeplabv3_resnet50")
        if environment.get("HOLOSPEX_INITIAL_CHECKPOINT_URI") or environment.get("HOLOSPEX_INITIAL_CHECKPOINT_SHA256"):
            raise ValueError("Backbone and initial checkpoints are mutually exclusive")
    return bool(uri)


def validate_optimization_environment(environment: dict[str, str]) -> None:
    """Reject bad experiment inputs before installing packages or downloading."""
    schedule = environment.get("HOLOSPEX_LR_SCHEDULE")
    if schedule and schedule not in {"none", "cosine"}:
        raise ValueError("HOLOSPEX_LR_SCHEDULE must be none or cosine")
    warmup = environment.get("HOLOSPEX_WARMUP_EPOCHS")
    if warmup and (not warmup.isdigit() or not 0 <= int(warmup) < int(environment["HOLOSPEX_EPOCHS"])):
        raise ValueError("HOLOSPEX_WARMUP_EPOCHS must be an integer in [0, epochs)")
    for suffix in ("LR", "WEIGHT_DECAY", "BACKBONE_LR_MULTIPLIER", "MAX_DURATION_SECONDS", "DICE_WEIGHT", "LOVASZ_WEIGHT"):
        value = environment.get("HOLOSPEX_" + suffix)
        if not value:
            continue
        try:
            number = float(value)
        except ValueError as error:
            raise ValueError(f"HOLOSPEX_{suffix} must be numeric") from error
        allow_zero = suffix in {"WEIGHT_DECAY", "DICE_WEIGHT", "LOVASZ_WEIGHT"}
        if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
            raise ValueError(f"HOLOSPEX_{suffix} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    auxiliary_weight = environment.get("HOLOSPEX_AUXILIARY_LOSS_WEIGHT")
    if auxiliary_weight:
        try:
            number = float(auxiliary_weight)
        except ValueError as error:
            raise ValueError("HOLOSPEX_AUXILIARY_LOSS_WEIGHT must be numeric") from error
        if not math.isfinite(number) or not 0 <= number <= 10:
            raise ValueError("HOLOSPEX_AUXILIARY_LOSS_WEIGHT must be finite and in [0, 10]")


def validate_deadline(value: str, *, now: datetime | None = None) -> datetime:
    """An absolute deadline must survive time spent queued by Vertex."""
    try:
        deadline = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("HOLOSPEX_DEADLINE_UTC must be an aware ISO timestamp") from error
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("HOLOSPEX_DEADLINE_UTC must include a timezone")
    deadline = deadline.astimezone(timezone.utc)
    if deadline <= (now or datetime.now(timezone.utc)):
        raise ValueError("HOLOSPEX_DEADLINE_UTC has expired; refusing worker startup")
    return deadline


def verify_runtime() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError("The cloud runtime requires Python 3.11 or newer")
    # The image digest is pinned, and the package resolver must preserve this
    # CUDA wheel pair instead of choosing newer wheels during installation.
    for package, expected in (("torch", "2.8.0"), ("torchvision", "0.23.0")):
        actual = importlib.metadata.version(package).split("+", 1)[0]
        if actual != expected:
            raise RuntimeError(f"Expected {package} {expected}; found {actual}")
    import torch
    import torchvision  # noqa: F401; check compiled operator compatibility.

    if not torch.cuda.is_available():
        raise RuntimeError("Vertex runtime has no usable CUDA GPU")
    print(f"Runtime ready: Python {sys.version.split()[0]}, torch {torch.__version__}, "
          f"CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}", flush=True)


def main() -> None:
    environment = dict(os.environ)
    required = ("SOURCE_URI", "SOURCE_SHA256", "DATA_URI", "DATA_SHA256",
                "OUTPUT_URI", "RUN_NAME", "EPOCHS")
    for suffix in required:
        if not environment.get("HOLOSPEX_" + suffix):
            raise ValueError(f"Missing HOLOSPEX_{suffix}")
    for suffix in ("SOURCE_URI", "DATA_URI", "OUTPUT_URI"):
        split_gcs_uri(environment["HOLOSPEX_" + suffix])
    for suffix in ("SOURCE_SHA256", "DATA_SHA256"):
        if re.fullmatch(r"[0-9a-fA-F]{64}", environment["HOLOSPEX_" + suffix]) is None:
            raise ValueError(f"Invalid HOLOSPEX_{suffix}")
    if not environment["HOLOSPEX_EPOCHS"].isdigit() or int(environment["HOLOSPEX_EPOCHS"]) < 1:
        raise ValueError("HOLOSPEX_EPOCHS must be a positive integer")
    for suffix in ("WIDTH", "HEIGHT"):
        value = environment.get("HOLOSPEX_" + suffix)
        if value and (not value.isdigit() or int(value) < 1):
            raise ValueError(f"HOLOSPEX_{suffix} must be a positive integer")
    has_prepared = prepared_bundle_enabled(environment)
    has_initial_checkpoint = initial_checkpoint_enabled(environment)
    has_backbone_checkpoint = backbone_checkpoint_enabled(environment)
    validate_optimization_environment(environment)
    if environment.get("HOLOSPEX_DEADLINE_UTC"):
        validate_deadline(environment["HOLOSPEX_DEADLINE_UTC"])

    verify_runtime()
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "google-cloud-storage>=2.19,<4"], check=True, timeout=600)
    from google.cloud import storage

    # Vertex's project environment is explicit because ADC may otherwise pick
    # a Google-managed tenant project. No local gcloud login is needed here.
    project = environment.get("HOLOSPEX_PROJECT") or environment.get("CLOUD_ML_PROJECT_ID")
    client = storage.Client(project=project)
    work = Path("/work")
    work.mkdir(parents=True, exist_ok=True)
    bundles = [("SOURCE", "holospex"), ("DATA", "endoscapes")]
    if has_prepared:
        bundles.append(("PREPARED", "prepared"))
    for prefix, root in bundles:
        bucket, name = split_gcs_uri(environment[f"HOLOSPEX_{prefix}_URI"])
        archive = work / (prefix.lower() + ".tar.gz")
        client.bucket(bucket).blob(name).download_to_filename(str(archive), timeout=120)
        digest = verify_sha256(archive, environment[f"HOLOSPEX_{prefix}_SHA256"])
        safe_extract(archive, work, root)
        print(f"Verified and extracted {prefix.lower()} bundle: {digest}", flush=True)

    if has_initial_checkpoint:
        bucket, name = split_gcs_uri(environment["HOLOSPEX_INITIAL_CHECKPOINT_URI"])
        initial_path = work / "initial.pt"
        client.bucket(bucket).blob(name).download_to_filename(str(initial_path), timeout=120)
        verify_sha256(initial_path, environment["HOLOSPEX_INITIAL_CHECKPOINT_SHA256"])
    if has_backbone_checkpoint:
        bucket, name = split_gcs_uri(environment["HOLOSPEX_BACKBONE_CHECKPOINT_URI"])
        backbone_path = work / "backbone.pt"
        client.bucket(bucket).blob(name).download_to_filename(str(backbone_path), timeout=120)
        verify_sha256(backbone_path, environment["HOLOSPEX_BACKBONE_CHECKPOINT_SHA256"])

    constraints = work / "cuda-constraints.txt"
    constraints.write_text("torch==2.8.0\ntorchvision==0.23.0\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--constraint", str(constraints), "-e", str(work / "holospex/ml") + "[train]"],
                   check=True, timeout=900)
    verify_runtime()
    os.chdir(work / "holospex")
    command = entry_command(environment, work)
    if environment.get("HOLOSPEX_DEADLINE_UTC"):
        validate_deadline(environment["HOLOSPEX_DEADLINE_UTC"])
    # Replace bootstrap so the artifact runner receives termination signals.
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
