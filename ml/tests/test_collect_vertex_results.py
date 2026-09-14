"""Local result collection: checksummed downloads, safe paths, complete publish."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("collect_vertex_results", Path(__file__).resolve().parents[1] / "cloud/collect_vertex_results.py")
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)
COMPLETION = "gs://test-bucket/runs/control/attempts/a/status/completed.json"


class FakeGcloud:
    def __init__(self, artifacts=None):
        self.objects = {}
        self.commands = []
        self.timeouts = []
        self.manifest = {"state": "completed", "artifacts": {}, "run_name": "control"}
        for name, payload in (artifacts or {"train/best.pt": b"checkpoint", "manifest.json": b'{"root":"/work/endoscapes"}'}).items():
            self.manifest["artifacts"][name] = self.reference(name, payload)
        self.manifest["last_epoch_manifest"] = self.reference("epoch.json", b'{"epoch":12,"best_epoch":9}')
        self.before_download = None

    def reference(self, name, payload):
        uri = "gs://test-bucket/objects/" + name
        self.objects[uri] = payload
        return {"uri": uri, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    def __call__(self, command, *, check, capture_output, timeout=None):
        self.commands.append(command)
        self.timeouts.append(timeout)
        assert check and capture_output
        assert command[:3] == ["gcloud", "--configuration=holospex", "storage"]
        if command[3] == "cat":
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(self.manifest).encode())
        assert command[3] == "cp"
        if self.before_download:
            self.before_download()
        Path(command[5]).write_bytes(self.objects[command[4]])
        return subprocess.CompletedProcess(command, 0, stdout=b"")


class CollectVertexResultsTests(unittest.TestCase):
    def test_total_budget_shrinks_across_metadata_and_artifact_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            cloud, output, elapsed = FakeGcloud(), Path(temporary) / "out", [0.0]

            def download(command, **kwargs):
                result = cloud(command, **kwargs)
                elapsed[0] += 2
                return result

            with patch.object(collector.time, "monotonic", side_effect=lambda: elapsed[0]):
                collector.collect(COMPLETION, output, run=download, timeout_seconds=10)
            self.assertEqual(cloud.timeouts, [10, 8, 6, 4])
            self.assertEqual((output / "train/best.pt").read_bytes(), b"checkpoint")

    def test_download_timeout_cleans_stage_and_allows_verified_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            cloud, output = FakeGcloud(), Path(temporary) / "out"

            def stalled(command, **kwargs):
                if command[3] == "cp":
                    Path(command[5]).write_bytes(b"partial")
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                return cloud(command, **kwargs)

            with self.assertRaises(subprocess.TimeoutExpired):
                collector.collect(COMPLETION, output, run=stalled, timeout_seconds=10)
            self.assertEqual(list(Path(temporary).iterdir()), [])
            collector.collect(COMPLETION, output, run=cloud, timeout_seconds=10)
            self.assertEqual((output / "train/best.pt").read_bytes(), b"checkpoint")
            self.assertFalse(json.loads((output / "cloud-collection.json").read_text())["metadata_paths_rewritten"])

    def test_expired_total_budget_never_publishes_even_after_last_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            cloud, output, elapsed = FakeGcloud(), Path(temporary) / "out", [0.0]

            def finish_at_limit(command, **kwargs):
                result = cloud(command, **kwargs)
                if command[3] == "cp" and command[5].endswith("cloud-last-epoch.json"):
                    elapsed[0] = 10
                return result

            with patch.object(collector.time, "monotonic", side_effect=lambda: elapsed[0]), self.assertRaises(TimeoutError):
                collector.collect(COMPLETION, output, run=finish_at_limit, timeout_seconds=10)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_collect_verifies_files_and_preserves_cloud_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "completed"
            cloud = FakeGcloud()
            receipt = collector.collect(COMPLETION, output, run=cloud)
            self.assertEqual((output / "train/best.pt").read_bytes(), b"checkpoint")
            self.assertEqual(json.loads((output / "manifest.json").read_text())["root"], "/work/endoscapes")
            self.assertEqual(json.loads((output / "cloud-completion.json").read_text()), cloud.manifest)
            self.assertEqual(json.loads((output / "cloud-last-epoch.json").read_text())["epoch"], 12)
            self.assertFalse(receipt["metadata_paths_rewritten"])
            self.assertEqual(receipt["artifact_count"], 2)
            self.assertEqual(list(Path(temporary).iterdir()), [output])

    def test_checksum_or_size_failure_publishes_nothing(self):
        for replacement in (b"corruption", b"wrong length"):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as temporary:
                output, cloud = Path(temporary) / "completed", FakeGcloud()
                reference = cloud.manifest["artifacts"]["train/best.pt"]
                cloud.objects[reference["uri"]] = replacement
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    collector.collect(COMPLETION, output, run=cloud)
                self.assertFalse(output.exists())
                self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_existing_output_or_dangling_symlink_is_never_touched(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "completed"
            output.mkdir()
            marker = output / "user-data"
            marker.write_text("keep")
            cloud = FakeGcloud()
            with self.assertRaises(FileExistsError):
                collector.collect(COMPLETION, output, run=cloud)
            self.assertEqual(marker.read_text(), "keep")
            self.assertEqual(cloud.commands, [])
            link = Path(temporary) / "dangling"
            link.symlink_to(Path(temporary) / "absent")
            with self.assertRaises(FileExistsError):
                collector.collect(COMPLETION, link, run=cloud)
            self.assertTrue(link.is_symlink())

    def test_target_created_during_download_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, cloud = Path(temporary) / "completed", FakeGcloud()
            cloud.before_download = lambda: output.mkdir(exist_ok=True)
            with self.assertRaises(FileExistsError):
                collector.collect(COMPLETION, output, run=cloud)
            self.assertTrue(output.is_dir())
            self.assertEqual(list(Path(temporary).iterdir()), [output])

    def test_unsafe_paths_uris_and_noncompleted_manifests_fail_before_downloads(self):
        for name in ("../escape", "/absolute", "a/../b", "a//b", "a\\b", "cloud-completion.json", "cloud-collection.json/x"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                cloud = FakeGcloud()
                reference = cloud.manifest["artifacts"].pop("train/best.pt")
                cloud.manifest["artifacts"][name] = reference
                with self.assertRaises(ValueError):
                    collector.collect(COMPLETION, Path(temporary) / "out", run=cloud)
                self.assertEqual(len(cloud.commands), 1)
        for uri in ("https://example.com/checkpoint", "gs://bucket/../escape", "gs://bucket/model*", "gs://bucket/model?x", "gs://bucket", "gs://bucket/a%2Fb"):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                collector.validate_uri(uri)
        for state in ("running", "failed", None):
            with self.subTest(state=state):
                with self.assertRaisesRegex(ValueError, "completed"):
                    collector.validate_manifest({"state": state, "artifacts": {"x": {}}})

    def test_conflicting_paths_and_bad_reference_metadata_are_rejected(self):
        cloud = FakeGcloud()
        cloud.manifest["artifacts"]["train"] = cloud.manifest["artifacts"]["train/best.pt"]
        with self.assertRaisesRegex(ValueError, "parent directory"):
            collector.validate_manifest(cloud.manifest)
        for key, value in (("sha256", "wrong"), ("bytes", True), ("bytes", -1)):
            with self.subTest(key=key):
                cloud = FakeGcloud()
                cloud.manifest["artifacts"]["train/best.pt"][key] = value
                with self.assertRaises(ValueError):
                    collector.validate_manifest(cloud.manifest)

    def test_gcloud_failure_cleans_partial_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            cloud, output = FakeGcloud(), Path(temporary) / "out"

            def fail(command, **kwargs):
                if command[3] == "cp":
                    raise subprocess.CalledProcessError(1, command, stderr=b"permission denied")
                return cloud(command, **kwargs)

            with self.assertRaises(subprocess.CalledProcessError):
                collector.collect(COMPLETION, output, run=fail)
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
