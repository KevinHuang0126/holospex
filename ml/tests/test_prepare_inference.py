"""The cloud context carries selected weights and code, never arbitrary files."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "prepare_inference", Path(__file__).resolve().parents[1] / "cloud/prepare_inference.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "selected.pt"
        self.checkpoint.write_bytes(b"synthetic selected checkpoint")
        sha = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.enterContext(patch.object(MODULE.current_model, "CHECKPOINT_SHA256", sha))
        self.enterContext(patch.object(MODULE.current_model, "CHECKPOINT_BYTES", self.checkpoint.stat().st_size))
        selection = {"checkpointSha256": sha, "modelVersion": MODULE.current_model.MODEL_VERSION}
        files = {
            "ml/src/holospex_ml/live.py": "# selected source\n",
            "contracts/schemas/frame-result.schema.json": "{}",
            "ml/cloud/inference/Dockerfile": "FROM python:3.12-slim\n",
            "ml/cloud/inference/requirements.txt": "jsonschema\n",
            "ml/cloud/inference/start.sh": "#!/bin/sh\n",
            "ml/CURRENT_MODEL_SELECTION.json": json.dumps(selection),
            "ml/src/holospex_ml/.env": "PRIVATE_TOKEN=must-not-copy",
            "ml/data/frame.jpg": "must-not-copy",
        }
        for name, value in files.items():
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(value)
        self.enterContext(patch.object(MODULE.subprocess, "check_output", side_effect=lambda args, **kw:
            b"revision\n" if args[1] == "rev-parse" else
            b"ml/src/holospex_ml/live.py\0ml/src/holospex_ml/.env\0contracts/schemas/frame-result.schema.json\0"))

    def test_allowlist_hashes_and_existing_context_are_preserved(self):
        output = self.root / "context"
        result = MODULE.prepare(output, self.checkpoint, self.root)
        self.assertFalse((output / "ml/src/holospex_ml/.env").exists())
        self.assertFalse((output / "ml/data").exists())
        self.assertEqual(result["checkpointSha256"], MODULE.current_model.CHECKPOINT_SHA256)
        for name, record in result["files"].items():
            data = (output / name).read_bytes()
            self.assertEqual(len(data), record["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
        with self.assertRaisesRegex(ValueError, "new output"):
            MODULE.prepare(output, self.checkpoint, self.root)

    def test_wrong_checkpoint_fails_before_creating_context(self):
        self.checkpoint.write_bytes(b"wrong checkpoint")
        with self.assertRaises(ValueError):
            MODULE.prepare(self.root / "context", self.checkpoint, self.root)
        self.assertFalse((self.root / "context").exists())

    def test_symlinked_source_is_rejected(self):
        source = self.root / "ml/src/holospex_ml/live.py"
        source.unlink()
        source.symlink_to(self.root / "ml/src/holospex_ml/.env")
        with self.assertRaisesRegex(ValueError, "regular source"):
            MODULE.prepare(self.root / "context", self.checkpoint, self.root)
        self.assertFalse((self.root / "context").exists())


if __name__ == "__main__":
    unittest.main()
