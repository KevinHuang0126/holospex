import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "vertex_bootstrap", Path(__file__).resolve().parents[1] / "cloud/vertex_bootstrap.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class VertexBootstrapTests(unittest.TestCase):
    def archive(self, directory, members):
        path = directory / "source.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, kind in members:
                info = tarfile.TarInfo(name)
                if kind == "file":
                    info.size = 5
                    archive.addfile(info, io.BytesIO(b"hello"))
                else:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "../outside"
                    archive.addfile(info)
        return path

    def test_verified_bundle_extracts_without_replacing_existing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.archive(root, [("holospex/ml/file.txt", "file")])
            bootstrap.safe_extract(path, root / "work", "holospex")
            self.assertEqual((root / "work/holospex/ml/file.txt").read_text(), "hello")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                bootstrap.safe_extract(path, root / "work", "holospex")

    def test_unsafe_members_rejected_before_any_files_written(self):
        for invalid in (("holospex/../../outside", "file"),
                        ("/holospex/absolute", "file"),
                        ("other/file", "file"), ("holospex/link", "link")):
            with self.subTest(member=invalid), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = self.archive(root, [("holospex/good", "file"), invalid])
                with self.assertRaises(ValueError):
                    bootstrap.safe_extract(path, root / "work", "holospex")
                self.assertFalse((root / "work/holospex").exists())

    def test_source_checksum_mismatch_is_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.tar.gz"
            path.write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                bootstrap.verify_sha256(path, "0" * 64)
            with self.assertRaisesRegex(ValueError, "64-character"):
                bootstrap.verify_sha256(path, "not-a-sha")

    def test_gcs_uris_are_objects_not_http_or_signed_urls(self):
        self.assertEqual(bootstrap.split_gcs_uri("gs://bucket/releases/source.tar.gz"),
                         ("bucket", "releases/source.tar.gz"))
        for invalid in ("https://bucket/file", "gs://bucket", "gs://bucket/file?token=secret"):
            with self.subTest(uri=invalid), self.assertRaises(ValueError):
                bootstrap.split_gcs_uri(invalid)

    def test_runner_arguments_preserve_spaces_and_only_requested_experiments(self):
        environment = {"HOLOSPEX_OUTPUT_URI": "gs://bucket/run outputs",
                       "HOLOSPEX_RUN_NAME": "cloud-001", "HOLOSPEX_EPOCHS": "12"}
        command = bootstrap.entry_command(environment, Path("/work"))
        self.assertIn("gs://bucket/run outputs", command)
        self.assertNotIn("--augmentation", command)
        environment["HOLOSPEX_AUGMENTATION"] = "paired"
        command = bootstrap.entry_command(environment, Path("/work"))
        self.assertEqual(command[-2:], ["--augmentation", "paired"])

    def test_dice_configuration_reaches_hyphenated_runner_flags(self):
        environment = {"HOLOSPEX_OUTPUT_URI": "gs://bucket/runs/dice",
                       "HOLOSPEX_RUN_NAME": "dice", "HOLOSPEX_EPOCHS": "40",
                       "HOLOSPEX_LOSS": "ce_generalized_dice", "HOLOSPEX_DICE_WEIGHT": "1.0"}
        command = bootstrap.entry_command(environment, Path("/work"))
        self.assertEqual(command[-4:], ["--loss", "ce_generalized_dice", "--dice-weight", "1.0"])

    def test_resolution_configuration_reaches_runner(self):
        environment = {"HOLOSPEX_OUTPUT_URI": "gs://bucket/runs/high-resolution",
                       "HOLOSPEX_RUN_NAME": "high-resolution", "HOLOSPEX_EPOCHS": "40",
                       "HOLOSPEX_WIDTH": "896", "HOLOSPEX_HEIGHT": "512"}
        command = bootstrap.entry_command(environment, Path("/work"))
        self.assertEqual(command[-4:], ["--width", "896", "--height", "512"])

    def test_prepared_bundle_requires_uri_and_checksum_and_reaches_runner(self):
        environment = {"HOLOSPEX_OUTPUT_URI": "gs://bucket/runs/reviewed",
                       "HOLOSPEX_RUN_NAME": "reviewed", "HOLOSPEX_EPOCHS": "40"}
        self.assertFalse(bootstrap.prepared_bundle_enabled(environment))
        environment["HOLOSPEX_PREPARED_URI"] = "gs://bucket/prepared.tar.gz"
        with self.assertRaisesRegex(ValueError, "Set both"):
            bootstrap.entry_command(environment, Path("/work"))
        environment["HOLOSPEX_PREPARED_SHA256"] = "0" * 64
        self.assertTrue(bootstrap.prepared_bundle_enabled(environment))
        command = bootstrap.entry_command(environment, Path("/work"))
        self.assertEqual(command[-2:], ["--prepared-package", "/work/prepared/package.json"])
        environment["HOLOSPEX_PREPARED_SHA256"] = "invalid"
        with self.assertRaisesRegex(ValueError, "Invalid"):
            bootstrap.prepared_bundle_enabled(environment)


if __name__ == "__main__":
    unittest.main()
