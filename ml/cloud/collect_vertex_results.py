"""Collect one completed Vertex attempt locally through the authenticated gcloud CLI.

Artifacts and their original metadata are preserved verbatim. In particular,
the cloud training manifest's absolute dataset paths are not rewritten here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlsplit


RESERVED = {"cloud-completion.json", "cloud-last-epoch.json", "cloud-collection.json"}


def validate_uri(uri):
    if not isinstance(uri, str):
        raise ValueError("Artifact URI must be a string.")
    parsed = urlsplit(uri)
    parts = parsed.path[1:].split("/")
    if (parsed.scheme != "gs" or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", parsed.netloc)
            or parsed.query or parsed.fragment or not parsed.path.startswith("/")
            or any(not part or part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in parts)):
        raise ValueError("Require an exact gs://bucket/object URI without wildcards or traversal.")
    return uri


def validate_name(name):
    if (not isinstance(name, str) or not name or name.startswith("/") or "\\" in name
            or any(not part or part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in name.split("/"))):
        raise ValueError(f"Unsafe logical artifact path: {name!r}")
    if name.split("/", 1)[0] in RESERVED:
        raise ValueError(f"Artifact collides with collector metadata: {name!r}")
    return name


def validate_reference(value):
    if not isinstance(value, dict):
        raise ValueError("Artifact reference must be an object.")
    validate_uri(value.get("uri"))
    digest = value.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[a-fA-F0-9]{64}", digest) is None:
        raise ValueError("Every artifact requires its SHA-256 digest.")
    size = value.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("Every artifact requires a nonnegative integer byte count.")


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("state") != "completed":
        raise ValueError("Only a completed attempt can be collected as complete.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Completion record requires a nonempty artifact map.")
    for name, reference in artifacts.items():
        validate_name(name)
        validate_reference(reference)
        if any(str(parent) in artifacts for parent in PurePosixPath(name).parents if str(parent) != "."):
            raise ValueError(f"An artifact is also a parent directory: {name!r}")
    if manifest.get("last_epoch_manifest") is not None:
        validate_reference(manifest["last_epoch_manifest"])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_new_target(path):
    # lexists also rejects dangling symlinks, which Path.exists would miss.
    if os.path.lexists(path):
        raise FileExistsError(f"Output already exists; choose a new directory: {path}")


def collect(completion_uri, output_dir, *, configuration="holospex", run=subprocess.run, timeout_seconds=None):
    """Publish a complete verified collection within an optional total budget.

    The budget covers the whole collection, not each individual download.
    Timeout leaves no published directory, so a later collection can retry.
    """
    validate_uri(completion_uri)
    if not isinstance(configuration, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", configuration):
        raise ValueError("Configuration must be a gcloud configuration name.")
    output = Path(output_dir).absolute()
    ensure_new_target(output)
    if timeout_seconds is not None and (type(timeout_seconds) not in (int, float)
                                       or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("Collection timeout_seconds must be finite and positive")
    deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None

    def remaining():
        if deadline is None:
            return None
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Artifact collection exhausted its total time budget")
        return seconds

    def fetch(command):
        seconds = remaining()
        options = {"timeout": seconds} if seconds is not None else {}
        return run(command, check=True, capture_output=True, **options)

    base = ["gcloud", f"--configuration={configuration}", "storage"]
    fetched = fetch(base + ["cat", completion_uri])
    payload = fetched.stdout
    manifest = json.loads(payload)
    validate_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.collect-", dir=output.parent))
    try:
        (stage / "cloud-completion.json").write_bytes(payload)
        downloads = list(manifest["artifacts"].items())
        if manifest.get("last_epoch_manifest") is not None:
            downloads.append(("cloud-last-epoch.json", manifest["last_epoch_manifest"]))
        for name, reference in downloads:
            destination = stage.joinpath(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fetch(base + ["cp", reference["uri"], str(destination)])
            if not destination.is_file() or destination.is_symlink():
                raise ValueError(f"Download did not produce a regular file: {name}")
            if destination.stat().st_size != reference["bytes"]:
                raise ValueError(f"Byte-count mismatch for {name}")
            if sha256(destination) != reference["sha256"].lower():
                raise ValueError(f"SHA-256 mismatch for {name}")
        receipt = {
            "completion_uri": completion_uri,
            "completion_sha256": hashlib.sha256(payload).hexdigest(),
            "completion_bytes": len(payload),
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "artifact_count": len(manifest["artifacts"]),
            "metadata_paths_rewritten": False,
        }
        (stage / "cloud-collection.json").write_text(json.dumps(receipt, indent=2) + "\n")
        # Check again after potentially long downloads before publishing a
        # complete directory on the same filesystem in one rename operation.
        remaining()
        ensure_new_target(output)
        stage.rename(output)
        return receipt
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-uri", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configuration", default="holospex")
    args = parser.parse_args(argv)
    try:
        receipt = collect(args.completion_uri, args.output_dir, configuration=args.configuration)
    except (ValueError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Collection failed: {error}\n")
    print(f"Verified and collected {receipt['artifact_count']} artifacts into {args.output_dir.absolute()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
