"""Freeze allowed source and test the extracted bundle before issuing a receipt.

This utility never uploads objects, creates jobs, or changes controller state.
Use a fresh output directory for every version, including after a failed check.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import tempfile


BUCKET = "gs://eastwest72hack26bos-501-holospex-ml"
ARCHIVE_NAME = "holospex-autonomous-source.tar.gz"
PATTERNS = (
    "ml/src/**/*.py", "ml/tests/**/*.py", "ml/cloud/*.py", "ml/cloud/*.sh", "ml/cloud/*.yaml",
    "ml/review/*.py", "ml/review/*.md", "ml/review/*.html", "ml/review/*.js", "ml/review/*.css",
    "ml/*.md", "ml/pyproject.toml", "contracts/**/*.json", "contracts/*.md", "assets/demo/**/*.json",
    "AGENTS.md", "README.md", "docs/architecture.md",
)
EXCLUDED_PARTS = {
    "data", "datasets", "outputs", "weights", "checkpoints", "recordings", "credentials", "secrets",
    "node_modules", "__pycache__",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_demo(path: Path) -> bool:
    """Only include demo JSON with explicit, exclusively synthetic provenance."""
    sources = []

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "source":
                    sources.append(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(json.loads(path.read_text()))
    return bool(sources) and all(source == "synthetic_mock" for source in sources)


def source_files(repo_root: Path) -> list[Path]:
    root = repo_root.resolve()
    selected = []
    for path in sorted({path for pattern in PATTERNS for path in root.glob(pattern)}):
        relative = path.relative_to(root)
        parts = [part.lower() for part in relative.parts]
        if any(part.startswith(".") or part in EXCLUDED_PARTS for part in parts):
            continue
        name = parts[-1]
        if ("credential" in name or "client_secret" in name
                or "service_account" in name or "service-account" in name):
            continue
        if any(root.joinpath(*relative.parts[:i]).is_symlink() for i in range(1, len(relative.parts) + 1)):
            raise ValueError(f"Source symlink is not allowed: {relative}")
        if not path.is_file():
            continue
        if relative.parts[:2] == ("assets", "demo") and not _synthetic_demo(path):
            continue
        selected.append(path)
    if not selected:
        raise ValueError("No allowed source files found")
    return selected


def _normalized_member(info: tarfile.TarInfo) -> tarfile.TarInfo:
    if not info.isfile():
        raise ValueError(f"Only ordinary source files can be archived: {info.name}")
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.mode & stat.S_IXUSR else 0o644
    info.pax_headers = {}
    return info


def package_source(repo_root: Path, output_dir: Path) -> dict:
    root, output = repo_root.resolve(), output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / ARCHIVE_NAME
    receipt_path = output / "source-bundle.json"
    log_path = output / "bundle-tests.log"
    if any(path.exists() for path in (archive_path, receipt_path, log_path)):
        raise FileExistsError("Immutable package output already exists; use a new version directory")
    python = root / ".venv/bin/python"
    if not python.is_file():
        raise ValueError(f"Repository test interpreter is missing: {python}")
    paths = source_files(root)
    inventory = []
    # Exclusive creation also prevents two simultaneous packagers from sharing
    # one output. Zero gzip/tar timestamps make unchanged source reproducible.
    with archive_path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", dereference=True) as archive:
                for path in paths:
                    before = sha256(path)
                    size = path.stat().st_size
                    if path.relative_to(root).parts[:2] == ("assets", "demo") and not _synthetic_demo(path):
                        raise RuntimeError(f"Demo provenance changed during packaging: {path.relative_to(root)}")
                    name = "holospex/" + path.relative_to(root).as_posix()
                    archive.add(path, arcname=name, recursive=False, filter=_normalized_member)
                    if sha256(path) != before or path.stat().st_size != size:
                        raise RuntimeError(f"Concurrent source mutation: {path.relative_to(root)}")
                    inventory.append({"path": name, "sha256": before, "bytes": size})
    digest = sha256(archive_path)
    with tempfile.TemporaryDirectory(prefix="holospex-autonomous-bundle-") as temporary:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(temporary, filter="data")
        extracted = Path(temporary) / "holospex"
        # Verify archived bytes too: even a change reverted before the second
        # source hash must not allow an inconsistent snapshot to pass.
        for member in inventory:
            path = Path(temporary) / member["path"]
            if path.stat().st_size != member["bytes"] or sha256(path) != member["sha256"]:
                raise RuntimeError(f"Archived source mismatch: {member['path']}")
        command = [str(python), "-m", "unittest", "discover", "-s", "ml/tests", "-v"]
        environment = {**os.environ, "PYTHONPATH": str(extracted / "ml/src"), "PYTHONNOUSERSITE": "1"}
        environment.pop("PYTHONHOME", None)
        with log_path.open("x") as log:
            result = subprocess.run(command, cwd=extracted, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Extracted full ML suite failed; no source receipt issued. See {log_path}")
    if sha256(archive_path) != digest:
        raise RuntimeError("Source archive changed during verification; no receipt issued")
    source = {"path": str(archive_path), "sha256": digest, "bytes": archive_path.stat().st_size,
              "uri": f"{BUCKET}/bundles/{digest}/{archive_path.name}", "members": inventory,
              "verification": {"suite": "full ML unittest discovery from extracted bundle", "returncode": 0,
                               "log": str(log_path), "python": str(python)}}
    with receipt_path.open("x") as receipt:
        receipt.write(json.dumps(source, indent=2, allow_nan=False) + "\n")
    return source


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    source = package_source(args.repo_root, args.output_dir)
    print(json.dumps({key: value for key, value in source.items() if key != "members"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
