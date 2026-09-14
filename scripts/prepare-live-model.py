"""Explicitly install Person 1's pinned checkpoint; never download at startup.

Use --checkpoint for a supplied file or --url for an HTTPS download link.
Only verified bytes are published to the ignored current-model path. Private
download URLs are never logged or saved, and an existing different model is
preserved. This helper needs only Python's standard library.
"""
from __future__ import annotations

from argparse import ArgumentParser
from contextlib import closing
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml/src"))
from holospex_ml import current_model as selection  # noqa: E402

CHUNK_BYTES = 1024 * 1024
DOWNLOAD_SECONDS = 300
SOCKET_SECONDS = 30


class PreparationError(ValueError):
    pass


def https_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.fragment or parsed.port == 0
                or any(ord(char) <= 32 or ord(char) == 127 for char in value)):
            raise ValueError
    except ValueError:
        raise PreparationError("The checkpoint download requires a valid HTTPS URL without embedded credentials.") from None
    return value


class HttpsRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        # Signed links may redirect to object storage; never downgrade to HTTP.
        return super().redirect_request(request, response, code, message, headers, https_url(newurl))


def verify_current(path: Path) -> None:
    if path.is_symlink():
        raise PreparationError("The current checkpoint must be a regular file, not a symbolic link.")
    try:
        selection.verify_checkpoint(path)
    except (OSError, ValueError):
        raise PreparationError("Checkpoint does not match Person 1's current model size and SHA-256; no model was replaced.") from None


def copy_bounded(source, target, *, deadline: float | None = None) -> None:
    total = 0
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise PreparationError("Checkpoint download exceeded its time limit; retry with a fresh download link.")
        # A single extra byte detects overlong responses without reading them all.
        chunk = source.read(min(CHUNK_BYTES, selection.CHECKPOINT_BYTES - total + 1))
        if deadline is not None and time.monotonic() >= deadline:
            raise PreparationError("Checkpoint download exceeded its time limit; retry with a fresh download link.")
        if not chunk:
            break
        total += len(chunk)
        if total > selection.CHECKPOINT_BYTES:
            raise PreparationError("Checkpoint exceeds the selected model's expected size.")
        target.write(chunk)
    if total != selection.CHECKPOINT_BYTES:
        raise PreparationError("Checkpoint is incomplete or has an unexpected size.")


def download(url: str, target) -> None:
    url = https_url(url)
    deadline = time.monotonic() + DOWNLOAD_SECONDS
    try:
        request = Request(url, headers={"Accept-Encoding": "identity"})
        with closing(build_opener(HttpsRedirects()).open(request, timeout=SOCKET_SECONDS)) as response:
            if response.status != 200:
                raise PreparationError("Checkpoint download did not return a complete file.")
            encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
            length = response.headers.get("Content-Length")
            if encoding != "identity":
                raise PreparationError("Checkpoint download must return uncompressed file bytes.")
            if length is not None and (not length.isascii() or not length.isdecimal()
                                       or int(length) != selection.CHECKPOINT_BYTES):
                raise PreparationError("Checkpoint download has an unexpected size.")
            copy_bounded(response, target, deadline=deadline)
    except PreparationError:
        raise
    except HTTPError as error:
        # Close failed responses too: their finalizer may otherwise log a URL.
        error.close()
        raise PreparationError("Checkpoint download failed. Check access or supply a fresh HTTPS download link.") from None
    except Exception:
        # urllib errors can embed a signed URL or response body. Keep them local.
        raise PreparationError("Checkpoint download failed. Check access or supply a fresh HTTPS download link.") from None


def prepare(*, checkpoint: Path | None = None, url: str | None = None,
            destination: Path | None = None) -> Path:
    if checkpoint is not None and url is not None:
        raise PreparationError("Supply either a local checkpoint or a download URL.")
    destination = selection.CHECKPOINT_PATH if destination is None else destination
    if destination.exists() or destination.is_symlink():
        verify_current(destination)
        return destination
    if checkpoint is None and url is None:
        raise PreparationError("Current checkpoint is missing. Supply --checkpoint PATH or --url HTTPS_URL; see ml/CURRENT_MODEL.md.")

    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".checkpoint-", suffix=".part", dir=destination.parent,
                                         delete=False) as target:
            temporary = Path(target.name)
            if checkpoint is not None:
                with checkpoint.open("rb") as source:
                    if os.fstat(source.fileno()).st_size != selection.CHECKPOINT_BYTES:
                        raise PreparationError("Supplied checkpoint has an unexpected size.")
                    copy_bounded(source, target)
            else:
                download(url, target)
            target.flush()
            os.fsync(target.fileno())
        verify_current(temporary)
        try:
            # Atomic publication without overwriting a file created concurrently.
            os.link(temporary, destination)
        except FileExistsError:
            verify_current(destination)
        return destination
    except OSError:
        raise PreparationError("Could not read or install the checkpoint. Check the supplied file and destination permissions; existing models are preserved.") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="An existing checkpoint supplied by Person 1.")
    source.add_argument("--url", help="An authorized HTTPS link to the exact checkpoint file.")
    args = parser.parse_args()
    try:
        result = prepare(checkpoint=args.checkpoint, url=args.url)
    except PreparationError as error:
        print(f"Model setup: {error}", file=sys.stderr)
        return 1
    print(f"Verified current checkpoint: {result}")
    print(f"Model version: {selection.MODEL_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
