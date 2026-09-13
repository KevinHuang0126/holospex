"""Fetch only Endoscapes-Seg50 pairs from the authors' public ZIP release.

Importing this module performs no network I/O. The public author release is
different from PhysioNet's account/DUA route; do not send credentials here.
Archive symlinks are resolved in memory and materialized as regular files.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import struct
import tempfile
import threading
import urllib.request
import zipfile
import zlib


OFFICIAL_ARCHIVE_URL = "https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip"
OFFICIAL_RELEASE_PAGE = "https://github.com/CAMMA-public/Endoscapes"
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_READ_BYTES = 32 * 1024 * 1024
MAX_TRANSFER_BYTES = 512 * 1024 * 1024
MAX_OUTPUT_BYTES = 512 * 1024 * 1024


class DownloadError(ValueError):
    """An unsafe, inconsistent, or unsupported download was rejected."""


class _TransferBudget:
    def __init__(self):
        self.used = 0
        self.lock = threading.Lock()

    def reserve(self, count):
        with self.lock:
            if self.used + count > MAX_TRANSFER_BYTES:
                raise DownloadError("Download exceeded the bounded transfer budget")
            self.used += count


class HTTPRangeReader(io.RawIOBase):
    """Seekable, bounded reader for zipfile; refuses a server's full response."""

    def __init__(self, url: str, *, opener=urllib.request.urlopen, metadata=None, budget=None):
        self.url = url
        self._opener = opener
        self.position = 0
        self.transferred_bytes = 0
        self._cache_start = 0
        self._cache = b""
        self._budget = budget or _TransferBudget()
        if metadata is None:
            with opener(urllib.request.Request(url, method="HEAD"), timeout=40) as response:
                self.size = int(response.headers.get("Content-Length", "0"))
                self.etag = response.headers.get("ETag")
        else:
            self.size, self.etag = metadata
        if self.size <= 0:
            raise DownloadError("Archive must advertise a positive Content-Length")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        bases = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        if whence not in bases or bases[whence] + offset < 0:
            raise DownloadError("Invalid archive seek")
        self.position = bases[whence] + offset
        return self.position

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size - self.position)
        if size <= 0:
            return b""
        if size > MAX_READ_BYTES:
            raise DownloadError("Archive read exceeds the per-request size limit")
        chunks = []
        remaining = size
        while remaining:
            cache_offset = self.position - self._cache_start
            if not 0 <= cache_offset < len(self._cache):
                count = min(max(remaining, 65536), self.size - self.position)
                self._budget.reserve(count)
                end = self.position + count - 1
                headers = {"Range": f"bytes={self.position}-{end}", "Accept-Encoding": "identity"}
                if self.etag:
                    headers["If-Match"] = self.etag
                request = urllib.request.Request(self.url, headers=headers)
                with self._opener(request, timeout=40) as response:
                    expected = f"bytes {self.position}-{end}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise DownloadError("Server did not honor the exact bounded HTTP Range")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise DownloadError("Encoded HTTP range responses are unsupported")
                    if self.etag and response.headers.get("ETag", self.etag) != self.etag:
                        raise DownloadError("Archive changed during download")
                    data = response.read(count + 1)
                    if len(data) != count:
                        raise DownloadError("Partial response has an unexpected length")
                self.transferred_bytes += count
                self._cache_start, self._cache = self.position, data
                cache_offset = 0
            take = min(remaining, len(self._cache) - cache_offset)
            chunks.append(self._cache[cache_offset:cache_offset + take])
            self.position += take
            remaining -= take
        return b"".join(chunks)


def _safe_archive_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts:
        raise DownloadError(f"Unsafe archive path: {name!r}")
    if not path.parts or path.parts[0] != "endoscapes":
        raise DownloadError(f"Archive entry is outside the expected root: {name!r}")
    return str(path)


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    if info.flag_bits & 1:
        raise DownloadError("Encrypted archive members are unsupported")
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise DownloadError("Unsupported ZIP compression")
    if info.file_size > MAX_MEMBER_BYTES or info.compress_size > MAX_READ_BYTES:
        raise DownloadError(f"Archive member exceeds the size limit: {info.filename}")
    # zipfile verifies the decompressed CRC. A bounded read also rejects members
    # whose actual decompressed size disagrees with the directory metadata.
    with archive.open(info) as stream:
        data = stream.read(info.file_size + 1)
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise DownloadError(f"Archive member failed size/CRC validation: {info.filename}")
    return data


def _read_member_independent(reader: HTTPRangeReader, info: zipfile.ZipInfo) -> bytes:
    """Read one bounded local record without sharing ZipFile's seek lock."""
    if info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise DownloadError("Encrypted/unsupported compression in archive member")
    if info.file_size > MAX_MEMBER_BYTES or info.compress_size > MAX_READ_BYTES:
        raise DownloadError("Archive member exceeds the size limit")
    reader.seek(info.header_offset)
    header = reader.read(30)
    if len(header) != 30:
        raise DownloadError("Truncated ZIP local header")
    signature, _, flags, method, _, _, _, _, _, name_length, extra_length = struct.unpack("<4s5H3L2H", header)
    if signature != b"PK\x03\x04" or flags != info.flag_bits or method != info.compress_type:
        raise DownloadError("ZIP local header disagrees with the central directory")
    filename = reader.read(name_length).decode("utf-8" if flags & 0x800 else "cp437")
    if filename != info.filename:
        raise DownloadError("ZIP local filename disagrees with the central directory")
    reader.seek(extra_length, io.SEEK_CUR)
    compressed = reader.read(info.compress_size)
    if method == zipfile.ZIP_STORED:
        data = compressed
    else:
        decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        data = decoder.decompress(compressed, info.file_size + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise DownloadError("Invalid or oversized compressed ZIP member")
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise DownloadError(f"Archive member failed size/CRC validation: {info.filename}")
    return data


def resolve_member(archive: zipfile.ZipFile, name: str, *, link_cache=None) -> zipfile.ZipInfo:
    """Resolve file AND directory links without creating filesystem symlinks."""
    name = _safe_archive_name(name)
    link_cache = {} if link_cache is None else link_cache
    seen = set()
    for _ in range(12):
        if name in seen:
            raise DownloadError("Archive symlink cycle")
        seen.add(name)
        parts = PurePosixPath(name).parts
        changed = False
        for length in range(1, len(parts) + 1):
            prefix = "/".join(parts[:length])
            try:
                info = archive.getinfo(prefix)
            except KeyError:
                continue
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                if info.file_size > 4096:
                    raise DownloadError("Archive symlink target is too large")
                if prefix not in link_cache:
                    link_cache[prefix] = _read_member(archive, info).decode("utf-8")
                target = link_cache[prefix]
                if target.startswith("/") or "\\" in target or ":" in target:
                    raise DownloadError("Unsafe archive symlink target")
                name = _safe_archive_name(posixpath.normpath(posixpath.join(
                    posixpath.dirname(prefix), target, *parts[length:])))
                changed = True
                break
        if changed:
            continue
        try:
            info = archive.getinfo(name)
        except KeyError as error:
            raise DownloadError(f"Archive is missing {name}") from error
        mode = info.external_attr >> 16
        if info.is_dir() or (stat.S_IFMT(mode) and not stat.S_ISREG(mode)):
            raise DownloadError(f"Expected a regular archive member: {name}")
        return info
    raise DownloadError("Too many archive symlink hops")


def _seg50_member(archive: zipfile.ZipFile, relative: str, *, link_cache=None) -> zipfile.ZipInfo:
    """Use the release's common JPEG store directly for COCO-named frames.

    train_seg is itself a directory symlink, and its JPEGs are more symlinks.
    Resolving every alias serially would require hundreds of remote round trips
    before workers can start. The unique COCO basename identifies the regular
    image in all/; only unusual layouts need the generic link resolver.
    """
    if re.fullmatch(r"(?:train|val|test)_seg/[0-9]+_[0-9]+\.jpg", relative):
        canonical = "endoscapes/all/" + relative.rsplit("/", 1)[1]
        try:
            archive.getinfo(canonical)
        except KeyError:
            pass
        else:
            return resolve_member(archive, canonical, link_cache=link_cache)
    return resolve_member(archive, "endoscapes/" + relative, link_cache=link_cache)


def _destination(root: Path, relative: str) -> Path:
    safe = _safe_archive_name("endoscapes/" + relative)
    path = root.joinpath(*PurePosixPath(safe).parts[1:])
    if not path.resolve().is_relative_to(root):
        raise DownloadError("Output path escapes the destination")
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise DownloadError("Output paths may not contain symlinks")
        cursor = cursor.parent
    return path


def _write_verified(root: Path, relative: str, data: bytes) -> bool:
    """Resume verified existing files; never replace differing user content."""
    path = _destination(root, relative)
    if path.exists():
        if path.stat().st_size == len(data) and path.read_bytes() == data:
            return False
        raise DownloadError(f"Refusing to replace differing existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        # An exclusive hard link publishes the verified complete file without
        # clobbering a destination created by another process in the meantime.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def download_seg50(output_dir: Path, *, metadata_only: bool = False,
                   progress: Callable[[str], None] | None = None) -> dict:
    """Download the canonical 343/76/74 labeled pairs, preserving video splits.

    No authentication or legal agreement is submitted. The caller must use the
    author release under its published terms and retain LICENSE/source details.
    metadata_only still transfers the ZIP directory (~17 MB), but no imagery.
    """
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    emit = progress or (lambda message: None)
    written = skipped = output_bytes = 0
    split_counts = {}
    selected_entries = []
    emit("Inspecting the official archive directory (about 17 MB of metadata)")
    with HTTPRangeReader(OFFICIAL_ARCHIVE_URL) as reader, zipfile.ZipFile(reader) as archive:
        names = [info.filename for info in archive.infolist()]
        if len(names) != len(set(names)):
            raise DownloadError("Archive contains duplicate member names")
        resolved = {}
        link_cache = {}

        def resolve(relative):
            if relative not in resolved:
                resolved[relative] = _seg50_member(archive, relative, link_cache=link_cache)
            return resolved[relative]

        def record(relative, info, data):
            selected_entries.append({"path": relative, "archivePath": info.filename,
                                     "bytes": len(data), "crc32": f"{info.CRC:08x}",
                                     "sha256": hashlib.sha256(data).hexdigest()})

        def obtain(relative: str) -> bytes:
            nonlocal written, skipped, output_bytes
            info = resolve(relative)
            if output_bytes + info.file_size > MAX_OUTPUT_BYTES:
                raise DownloadError("Selected output exceeds the size budget")
            path = _destination(root, relative)
            if path.exists() and path.is_file() and path.stat().st_size == info.file_size:
                data = path.read_bytes()
                if zlib.crc32(data) == info.CRC:
                    skipped += 1
                    output_bytes += len(data)
                    record(relative, info, data)
                    return data
            data = _read_member(archive, info)
            created = _write_verified(root, relative, data)
            written += int(created)
            skipped += int(not created)
            output_bytes += len(data)
            record(relative, info, data)
            emit(f"Downloaded {relative}")
            return data

        for relative in ("LICENSE", "README.md", "seg_label_map.txt", "train_seg_vids.txt",
                         "val_seg_vids.txt", "test_seg_vids.txt", "train_vids.txt", "val_vids.txt", "test_vids.txt"):
            obtain(relative)

        pairs = []
        for split, expected in (("train", 343), ("val", 76), ("test", 74)):
            annotations = json.loads(obtain(f"{split}_seg/annotation_coco.json"))
            images = annotations.get("images", [])
            if len(images) != expected:
                raise DownloadError(f"Unexpected {split} segmentation count: {len(images)}; expected {expected}")
            split_counts[split] = len(images)
            for image in images:
                filename = image.get("file_name", "")
                if not re.fullmatch(r"[0-9]+_[0-9]+\.jpg", filename):
                    raise DownloadError(f"Unexpected image filename: {filename!r}")
                pairs.extend((f"{split}_seg/{filename}", f"semseg/{filename[:-4]}.png"))
        if not metadata_only:
            emit(f"Planning {len(pairs) // 2} labeled image/mask pairs from the archive index")
            # Nearby ZIP members share the reader cache. Sort by physical order
            # after resolving links, without changing the train/val/test split.
            pairs.sort(key=lambda name: resolve(name).header_offset)
            if output_bytes + sum(resolve(name).file_size for name in pairs) > MAX_OUTPUT_BYTES:
                raise DownloadError("Selected output exceeds the size budget")
            emit(f"Downloading/verifying {len(pairs)} image and mask files with 6 workers")
            local = threading.local()
            workers = []
            worker_lock = threading.Lock()

            def fetch(relative):
                info = resolve(relative)
                path = _destination(root, relative)
                if path.exists() and path.is_file() and path.stat().st_size == info.file_size:
                    data = path.read_bytes()
                    if zlib.crc32(data) == info.CRC:
                        return relative, info, data, False
                if not hasattr(local, "reader"):
                    local.reader = HTTPRangeReader(OFFICIAL_ARCHIVE_URL, metadata=(reader.size, reader.etag), budget=reader._budget)
                    with worker_lock:
                        workers.append(local.reader)
                data = _read_member_independent(local.reader, info)
                return relative, info, data, _write_verified(root, relative, data)

            try:
                with ThreadPoolExecutor(max_workers=6) as executor:
                    for completed, (relative, info, data, created) in enumerate(executor.map(fetch, pairs), 1):
                        written += int(created)
                        skipped += int(not created)
                        output_bytes += len(data)
                        record(relative, info, data)
                        if completed % 50 == 0 or completed == len(pairs):
                            emit(f"Image/mask files ready: {completed}/{len(pairs)}")
            finally:
                for worker in workers:
                    worker.close()
            reader.transferred_bytes += sum(worker.transferred_bytes for worker in workers)
            if reader.transferred_bytes > MAX_TRANSFER_BYTES:
                raise DownloadError("Download exceeded the aggregate transfer budget")
        selected_entries.sort(key=lambda item: item["path"])
        summary = {
            "sourceUrl": OFFICIAL_ARCHIVE_URL,
            "sourcePage": OFFICIAL_RELEASE_PAGE,
            "archiveBytes": reader.size,
            "archiveEtag": reader.etag,
            "downloadedAt": datetime.now(timezone.utc).isoformat(),
            "subset": "Endoscapes-Seg50",
            "metadataOnly": metadata_only,
            "splitImageCounts": split_counts,
            "filesWritten": written,
            "verifiedExistingFiles": skipped,
            "selectedBytes": output_bytes,
            "transferredBytes": reader.transferred_bytes,
            "license": "Author release: CC BY-NC-SA 4.0; see LICENSE and README.md",
            "selectedEntriesSha256": hashlib.sha256(json.dumps(selected_entries, sort_keys=True).encode()).hexdigest(),
            "selectedEntries": selected_entries,
        }
    # Unique run reports preserve previous download provenance when resuming.
    report_name = "download-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    _write_verified(root, report_name, (json.dumps(summary, indent=2) + "\n").encode())
    return summary
