import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from holospex_ml.download import (
    DownloadError, HTTPRangeReader, _read_member, _read_member_independent,
    _safe_archive_name, _seg50_member, _write_verified, resolve_member,
)


class Response(io.BytesIO):
    def __init__(self, body=b"", *, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


class MemoryHTTP:
    def __init__(self, body, *, honor_range=True, corrupt_range=False):
        self.body = body
        self.honor_range = honor_range
        self.corrupt_range = corrupt_range
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        headers = {"Content-Length": str(len(self.body)), "ETag": '"fixture"'}
        if request.get_method() == "HEAD":
            return Response(headers=headers)
        if not self.honor_range:
            return Response(self.body, headers=headers)
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        headers["Content-Range"] = f"bytes {start}-{end}/{len(self.body)}"
        if self.corrupt_range:
            headers["Content-Range"] = "bytes 0-0/1"
        return Response(self.body[start:end + 1], status=206, headers=headers)


def fixture_zip(entries):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data, symlink in entries:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = ((stat.S_IFLNK if symlink else stat.S_IFREG) | 0o644) << 16
            archive.writestr(info, data)
    return content.getvalue()


class RangeReaderTests(unittest.TestCase):
    def test_seek_and_cache_read_only_selected_bytes(self):
        http = MemoryHTTP(b"0123456789" * 20000)
        reader = HTTPRangeReader("https://example.test/archive.zip", opener=http)
        reader.seek(-20, io.SEEK_END)
        self.assertEqual(reader.read(5), b"01234")
        reader.seek(-10, io.SEEK_END)
        self.assertEqual(reader.read(), b"0123456789")
        self.assertEqual(reader.transferred_bytes, 20)
        self.assertEqual(len(http.requests), 2)  # HEAD plus one bounded GET

    def test_refuses_full_and_incorrect_range_responses(self):
        for options in ({"honor_range": False}, {"corrupt_range": True}):
            with self.subTest(options=options):
                reader = HTTPRangeReader("https://example.test/archive.zip", opener=MemoryHTTP(b"data", **options))
                with self.assertRaises(DownloadError):
                    reader.read(2)

    def test_refuses_unbounded_and_over_budget_downloads(self):
        http = MemoryHTTP(b"abc" * 100)
        reader = HTTPRangeReader("https://example.test/archive.zip", opener=http)
        with patch("holospex_ml.download.MAX_READ_BYTES", 5):
            with self.assertRaises(DownloadError):
                reader.read()
        with patch("holospex_ml.download.MAX_TRANSFER_BYTES", 5):
            with self.assertRaises(DownloadError):
                reader.read(2)
        self.assertEqual(len(http.requests), 1)


class ArchiveTests(unittest.TestCase):
    def test_canonical_jpeg_lookup_needs_no_remote_symlink_reads(self):
        payload = fixture_zip([
            ("endoscapes/train_seg", "25/train_2", True),
            ("endoscapes/25/train_2/4_25.jpg", "../../all/4_25.jpg", True),
            ("endoscapes/all/4_25.jpg", b"sample", False),
        ])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            with patch("holospex_ml.download._read_member", side_effect=AssertionError("Unexpected remote link read")):
                info = _seg50_member(archive, "train_seg/4_25.jpg")
                self.assertEqual(info.filename, "endoscapes/all/4_25.jpg")

    def test_jpeg_lookup_falls_back_for_layout_without_common_store(self):
        payload = fixture_zip([("endoscapes/train_seg/4_25.jpg", b"sample", False)])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            info = _seg50_member(archive, "train_seg/4_25.jpg")
            self.assertEqual(info.filename, "endoscapes/train_seg/4_25.jpg")

    def test_directory_symlink_targets_are_cached_across_members(self):
        payload = fixture_zip([
            ("endoscapes/train_seg", "25/train_2", True),
            ("endoscapes/25/train_2/4_25.jpg", b"first", False),
            ("endoscapes/25/train_2/4_50.jpg", b"second", False),
        ])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            cache = {}
            with patch("holospex_ml.download._read_member", wraps=_read_member) as read:
                resolve_member(archive, "endoscapes/train_seg/4_25.jpg", link_cache=cache)
                resolve_member(archive, "endoscapes/train_seg/4_50.jpg", link_cache=cache)
                self.assertEqual(read.call_count, 1)

    def test_resolves_directory_and_file_links_without_materializing_links(self):
        payload = fixture_zip([
            ("endoscapes/train_seg", "25/train_2", True),
            ("endoscapes/25/train_2/4_25.jpg", "../../all/4_25.jpg", True),
            ("endoscapes/all/4_25.jpg", b"sample", False),
        ])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            info = resolve_member(archive, "endoscapes/train_seg/4_25.jpg")
            self.assertEqual(info.filename, "endoscapes/all/4_25.jpg")
            self.assertEqual(_read_member(archive, info), b"sample")

    def test_rejects_escaping_links_and_cycles(self):
        for target in ("../../outside", "/tmp/outside", "loop"):
            with self.subTest(target=target):
                payload = fixture_zip([("endoscapes/loop", target, True)])
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    with self.assertRaises(DownloadError):
                        resolve_member(archive, "endoscapes/loop")

    def test_rejects_unsafe_member_paths(self):
        for path in ("/endoscapes/x", "endoscapes/../x", "endoscapes\\x", "C:/endoscapes/x", "other/x"):
            with self.subTest(path=path), self.assertRaises(DownloadError):
                _safe_archive_name(path)

    def test_independent_range_member_validates_data_and_crc(self):
        payload = fixture_zip([("endoscapes/all/1_30.jpg", b"sample" * 100, False)])
        reader = HTTPRangeReader("https://example.test/archive.zip", opener=MemoryHTTP(payload))
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            info = archive.infolist()[0]
            self.assertEqual(_read_member_independent(reader, info), b"sample" * 100)
            info.CRC ^= 1
            with self.assertRaises(DownloadError):
                _read_member_independent(reader, info)

    def test_refuses_compression_and_size_mismatches(self):
        payload = fixture_zip([("endoscapes/all/1_30.jpg", b"sample", False)])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            info = archive.infolist()[0]
            with patch("holospex_ml.download.MAX_MEMBER_BYTES", 2):
                with self.assertRaises(DownloadError):
                    _read_member(archive, info)
            info.compress_type = zipfile.ZIP_BZIP2
            with self.assertRaises(DownloadError):
                _read_member(archive, info)

    def test_existing_files_are_verified_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertTrue(_write_verified(root, "semseg/x.png", b"first"))
            self.assertFalse(_write_verified(root, "semseg/x.png", b"first"))
            with self.assertRaises(DownloadError):
                _write_verified(root, "semseg/x.png", b"different")
            self.assertEqual((root / "semseg/x.png").read_bytes(), b"first")

    def test_output_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "regular").mkdir()
            (root / "semseg").symlink_to(root / "regular", target_is_directory=True)
            with self.assertRaises(DownloadError):
                _write_verified(root, "semseg/x.png", b"sample")


if __name__ == "__main__":
    unittest.main()
