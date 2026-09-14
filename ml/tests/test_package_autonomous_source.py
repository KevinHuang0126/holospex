"""Source allowlist, immutable artifacts, and extracted-suite receipt gating."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("package_autonomous_source", Path(__file__).resolve().parents[1] / "cloud/package_autonomous_source.py")
packaging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packaging)


class SourcePackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / "repo"
        self.output = Path(self.temporary.name).resolve() / "version-one"
        for path, content in {
            ".venv/bin/python": "fixture interpreter",
            "ml/src/holospex_ml/model.py": "VALUE = 1\n",
            "ml/tests/test_fixture.py": "# fixture test\n",
            "ml/review/review.template.html": "<html>Review template</html>\n",
            "ml/cloud/vertex_bootstrap.py": "# bootstrap\n",
            "ml/pyproject.toml": "[project]\nname = 'fixture'\n",
            "contracts/schemas/frame-result.schema.json": "{}\n",
            "assets/demo/frame.json": '{"source":"synthetic_mock"}\n',
            "assets/demo/predicted.json": '{"source":"ml_prediction"}\n',
            "ml/outputs/private.py": "private output",
            "ml/data/private.py": "private dataset",
            "ml/src/weights/private.py": "private weight content",
            "contracts/credentials.json": '{"private_key":"do not include"}',
            ".git/config": "private remote",
        }.items():
            self.write(path, content)

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def run_success(self, output=None):
        def run(command, **kwargs):
            extracted = kwargs["cwd"]
            self.assertNotEqual(extracted, self.root)
            self.assertEqual(command[0], str(self.root / ".venv/bin/python"))
            self.assertEqual(command[1:], ["-m", "unittest", "discover", "-s", "ml/tests", "-v"])
            self.assertEqual(kwargs["env"]["PYTHONPATH"], str(extracted / "ml/src"))
            self.assertEqual(kwargs["env"]["PYTHONNOUSERSITE"], "1")
            self.assertTrue((extracted / "ml/review/review.template.html").is_file())
            self.assertFalse((self.output / "source-bundle.json").exists())
            return subprocess.CompletedProcess(command, 0)

        with patch.object(packaging.subprocess, "run", side_effect=run):
            return packaging.package_source(self.root, output or self.output)

    def test_includes_review_template_and_only_allowed_synthetic_source(self):
        source = self.run_success()
        names = {member["path"] for member in source["members"]}
        self.assertIn("holospex/ml/review/review.template.html", names)
        self.assertIn("holospex/assets/demo/frame.json", names)
        for forbidden in ("predicted.json", "private.py", "credentials.json", ".git/config", ".venv/bin/python"):
            self.assertFalse(any(name.endswith(forbidden) for name in names))
        with tarfile.open(source["path"]) as archive:
            self.assertEqual(set(archive.getnames()), names)
        self.assertEqual(packaging.sha256(Path(source["path"])), source["sha256"])
        self.assertEqual(source["uri"], f"{packaging.BUCKET}/bundles/{source['sha256']}/{packaging.ARCHIVE_NAME}")
        self.assertEqual(json.loads((self.output / "source-bundle.json").read_text()), source)

    def test_archive_is_reproducible_and_existing_version_is_never_overwritten(self):
        first = self.run_success()
        archive = Path(first["path"])
        before = archive.read_bytes()
        with patch.object(packaging.subprocess, "run") as run, self.assertRaises(FileExistsError):
            packaging.package_source(self.root, self.output)
        run.assert_not_called()
        self.assertEqual(archive.read_bytes(), before)
        # A different directory has identical compressed bytes for identical
        # content, independent of source filesystem timestamps.
        for path in packaging.source_files(self.root):
            path.touch()
        second_output = self.output.with_name("version-two")
        with patch.object(packaging.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            second = packaging.package_source(self.root, second_output)
        self.assertEqual(second["sha256"], first["sha256"])

    def test_failed_extracted_suite_does_not_publish_receipt(self):
        with patch.object(packaging.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)), \
                self.assertRaisesRegex(RuntimeError, "no source receipt"):
            packaging.package_source(self.root, self.output)
        self.assertFalse((self.output / "source-bundle.json").exists())
        self.assertTrue((self.output / packaging.ARCHIVE_NAME).is_file())
        self.assertTrue((self.output / "bundle-tests.log").is_file())

    def test_concurrent_source_mutation_prevents_tests_and_receipt(self):
        original = tarfile.TarFile.add
        target = self.root / "ml/src/holospex_ml/model.py"

        def mutate(archive, name, *args, **kwargs):
            result = original(archive, name, *args, **kwargs)
            if Path(name) == target:
                target.write_text("VALUE = 2\n")
            return result

        with patch.object(tarfile.TarFile, "add", new=mutate), patch.object(packaging.subprocess, "run") as run, \
                self.assertRaisesRegex(RuntimeError, "Concurrent source mutation"):
            packaging.package_source(self.root, self.output)
        run.assert_not_called()
        self.assertFalse((self.output / "source-bundle.json").exists())

    def test_source_symlink_cannot_export_files_outside_allowlist(self):
        external = self.root.parent / "private.py"
        external.write_text("private source")
        (self.root / "ml/src/holospex_ml/link.py").symlink_to(external)
        with self.assertRaisesRegex(ValueError, "symlink"):
            packaging.source_files(self.root)


if __name__ == "__main__":
    unittest.main()
