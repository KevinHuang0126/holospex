import importlib.util
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("prepare_live_model", ROOT / "scripts/prepare-live-model.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Response(BytesIO):
    def __init__(self, value, headers=None, status=200):
        super().__init__(value)
        self.headers = {} if headers is None else headers
        self.status = status


class PrepareLiveModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.payload = b"A small stand-in for the selected model's exact bytes."
        self.source = self.folder / "supplied.pt"
        self.source.write_bytes(self.payload)
        self.destination = self.folder / "weights/current/best.pt"
        self.enterContext(patch.object(MODULE.selection, "CHECKPOINT_BYTES", len(self.payload)))
        self.enterContext(patch.object(MODULE.selection, "CHECKPOINT_SHA256", sha256(self.payload).hexdigest()))
        self.enterContext(patch.object(MODULE.selection, "CHECKPOINT_PATH", self.destination))

    def assert_no_partial_checkpoint(self):
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.glob(".checkpoint-*")), [])

    def test_installs_exact_local_file_and_rerun_needs_no_source(self):
        self.assertEqual(MODULE.prepare(checkpoint=self.source), self.destination)
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertEqual(self.source.read_bytes(), self.payload)
        self.source.unlink()
        with patch.object(MODULE, "build_opener") as opener:
            self.assertEqual(MODULE.prepare(url="https://private.invalid/new-link"), self.destination)
            self.assertEqual(MODULE.prepare(), self.destination)
            opener.assert_not_called()
        self.assertEqual(list(self.destination.parent.glob(".checkpoint-*")), [])

    def test_wrong_same_size_hash_does_not_publish(self):
        self.source.write_bytes(b"X" * len(self.payload))
        with self.assertRaisesRegex(MODULE.PreparationError, "SHA-256"):
            MODULE.prepare(checkpoint=self.source)
        self.assert_no_partial_checkpoint()

    def test_existing_different_model_is_preserved_without_download(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b"old checkpoint")
        with patch.object(MODULE, "build_opener") as opener:
            with self.assertRaisesRegex(MODULE.PreparationError, "no model was replaced"):
                MODULE.prepare(url="https://private.invalid/new-link")
            opener.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b"old checkpoint")

    def test_concurrent_model_is_preserved_at_atomic_publication(self):
        def competing_publish(source, destination):
            destination.write_bytes(b"concurrent checkpoint")
            raise FileExistsError
        with patch.object(MODULE.os, "link", side_effect=competing_publish):
            with self.assertRaises(MODULE.PreparationError):
                MODULE.prepare(checkpoint=self.source)
        self.assertEqual(self.destination.read_bytes(), b"concurrent checkpoint")
        self.assertEqual(list(self.destination.parent.glob(".checkpoint-*")), [])

    def test_downloads_verified_bytes_without_saving_private_url(self):
        with patch.object(MODULE, "build_opener") as opener:
            response = Response(self.payload, {"Content-Length": str(len(self.payload))})
            opener.return_value.open.return_value = response
            MODULE.prepare(url="https://private.invalid/model?signature=SECRET")
            self.assertTrue(response.closed)
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_rejects_short_oversized_compressed_and_partial_downloads(self):
        examples = [
            Response(self.payload[:-1]), Response(self.payload + b"extra"),
            Response(self.payload, {"Content-Length": str(len(self.payload) + 1)}),
            Response(self.payload, {"Content-Length": "invalid"}),
            Response(self.payload, {"Content-Encoding": "gzip"}),
            Response(self.payload, status=206),
        ]
        for response in examples:
            with self.subTest(headers=response.headers, status=response.status):
                with patch.object(MODULE, "build_opener") as opener:
                    opener.return_value.open.return_value = response
                    with self.assertRaises(MODULE.PreparationError):
                        MODULE.prepare(url="https://private.invalid/model")
                self.assert_no_partial_checkpoint()
                self.assertTrue(response.closed)

    def test_download_is_size_and_time_bounded(self):
        response = Response(self.payload + b"X" * 1000)
        with self.assertRaisesRegex(MODULE.PreparationError, "exceeds"):
            MODULE.copy_bounded(response, BytesIO())
        self.assertEqual(response.tell(), len(self.payload) + 1)
        with patch.object(MODULE.time, "monotonic", return_value=2):
            with self.assertRaisesRegex(MODULE.PreparationError, "time limit"):
                MODULE.copy_bounded(BytesIO(self.payload), BytesIO(), deadline=1)
        target = BytesIO()
        with patch.object(MODULE.time, "monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(MODULE.PreparationError, "time limit"):
                MODULE.copy_bounded(BytesIO(self.payload), target, deadline=1)
        self.assertEqual(target.getvalue(), b"")

    def test_rejects_unsafe_urls_and_redirect_downgrade(self):
        for url in ("http://private.invalid/model", "https://user:secret@private.invalid/model",
                    "https://private.invalid/model#secret", "https://private.invalid:bad/model",
                    "https://private.invalid/model\nheader:secret"):
            with self.subTest(url=url), patch.object(MODULE, "build_opener") as opener:
                with self.assertRaises(MODULE.PreparationError):
                    MODULE.prepare(url=url)
                opener.assert_not_called()
                self.assert_no_partial_checkpoint()
        with self.assertRaises(MODULE.PreparationError):
            MODULE.HttpsRedirects().redirect_request(Request("https://private.invalid"), None,
                                                    302, "Found", {}, "http://insecure.invalid")

    def test_cli_redacts_signed_url_and_provider_error(self):
        private = "https://private.invalid/model?signature=SECRET"
        output = StringIO()
        error = HTTPError(private, 403, "Denied signature=SECRET", {}, BytesIO(b"SECRET"))
        with patch.object(MODULE, "build_opener") as opener:
            opener.return_value.open.side_effect = error
            with patch("sys.argv", ["prepare-live-model.py", "--url", private]), patch("sys.stderr", output):
                self.assertEqual(MODULE.main(), 1)
        self.assertIn("Checkpoint download failed", output.getvalue())
        self.assertNotIn("SECRET", output.getvalue())
        self.assertNotIn("private.invalid", output.getvalue())
        self.assertTrue(error.closed)
        self.assert_no_partial_checkpoint()

    def test_interrupted_download_cleans_partial_bytes_and_redacts_error(self):
        class BrokenResponse(Response):
            def read(self, size):
                if self.tell():
                    raise OSError("Read failed for https://private.invalid/?signature=SECRET")
                return super().read(4)
        response = BrokenResponse(self.payload)
        with patch.object(MODULE, "build_opener") as opener:
            opener.return_value.open.return_value = response
            with self.assertRaisesRegex(MODULE.PreparationError, "download failed") as caught:
                MODULE.prepare(url="https://private.invalid/?signature=SECRET")
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertTrue(response.closed)
        self.assert_no_partial_checkpoint()

    def test_missing_source_never_attempts_download(self):
        with patch.object(MODULE, "build_opener") as opener:
            with self.assertRaisesRegex(MODULE.PreparationError, "Supply --checkpoint"):
                MODULE.prepare()
            opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
