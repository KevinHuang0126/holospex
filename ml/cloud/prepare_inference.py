"""Build a small, checksum-bound container context without credentials or data.

The output must be a new directory. Only tracked Python source, wire schemas,
the explicit container files and the verified selected checkpoint are copied.
This performs no network requests, builds, deployment or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml/src"))
from holospex_ml import current_model  # noqa: E402


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(output: Path, checkpoint: Path, root: Path = ROOT) -> dict:
    output = output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing deployment inputs are preserved")
    current_model.verify_checkpoint(checkpoint)
    selected = json.loads((root / "ml/CURRENT_MODEL_SELECTION.json").read_text())
    if (selected["checkpointSha256"] != current_model.CHECKPOINT_SHA256
            or selected["modelVersion"] != current_model.MODEL_VERSION):
        raise ValueError("Selection receipt differs from the runtime pin")
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "ml/src/holospex_ml", "contracts/schemas"], cwd=root,
    ).decode().split("\0")
    files = [p for p in tracked if p and (
        p.startswith("ml/src/holospex_ml/") and p.endswith(".py")
        or p.startswith("contracts/schemas/") and p.endswith(".json"))]
    sources = {p: root / p for p in files}
    for name in ("Dockerfile", "requirements.txt", "start.sh"):
        sources[name] = root / "ml/cloud/inference" / name
    sources["model-selection.json"] = root / "ml/CURRENT_MODEL_SELECTION.json"
    sources["ml/weights/current/best.pt"] = checkpoint
    for source in sources.values():
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Expected a regular source file: {source}")
    output.mkdir(parents=True)
    for relative, source in sources.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o444)
    current_model.verify_checkpoint(output / "ml/weights/current/best.pt")
    manifest = {
        "sourceCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
        "modelVersion": current_model.MODEL_VERSION,
        "checkpointSha256": current_model.CHECKPOINT_SHA256,
        "files": {name: {"bytes": (output / name).stat().st_size,
                          "sha256": digest(output / name)} for name in sorted(sources)},
    }
    (output / "context-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=current_model.CHECKPOINT_PATH)
    args = parser.parse_args()
    result = prepare(args.output, args.checkpoint)
    print(f"Prepared {len(result['files'])} explicit files for {result['modelVersion']}")


if __name__ == "__main__":
    main()
