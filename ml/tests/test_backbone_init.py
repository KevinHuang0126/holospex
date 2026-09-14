"""Safe backbone initialization must preserve heads and fail before mutation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"), "Training dependencies not installed")
class BackboneInitializationTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "synthetic-backbone.pt"
        # Small structural fixture isolates initialization behavior without any
        # external weight download or large checkpoint writes in the suite.
        self.model = torch.nn.Module()
        self.model.backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))
        self.model.classifier = torch.nn.Sequential(torch.nn.Conv2d(4, 3, 1), torch.nn.BatchNorm2d(3))
        self.model.aux_classifier = torch.nn.Conv2d(4, 3, 1)
        state = {key: value.clone() + 1 for key, value in self.model.backbone.state_dict().items()}
        self.payload = {"format_version": 1, "artifact_type": "holospex_pretrained_backbone", "architecture": "resnet50",
                        "backbone_state": state, "provenance": {"source_url": "https://example.invalid/synthetic-fixture",
                                                                "license": "test fixture", "case_ids": [1, 2]}}
        self.before = {key: value.clone() for key, value in self.model.state_dict().items()}

    def write(self):
        import torch

        torch.save(self.payload, self.path)

    def assert_unchanged(self):
        import torch

        for key, original in self.before.items():
            self.assertTrue(torch.equal(original, self.model.state_dict()[key]), key)

    def test_exact_backbone_load_preserves_all_head_tensors_and_records_source(self):
        import torch
        from holospex_ml.model import load_backbone_checkpoint

        self.write()
        with patch("holospex_ml.model.torch.load", wraps=torch.load) as loader:
            provenance = load_backbone_checkpoint(self.model, self.path, architecture="deeplabv3_resnet50")
        loader.assert_called_once_with(self.path.resolve(), map_location="cpu", weights_only=True)
        for key, value in self.payload["backbone_state"].items():
            self.assertTrue(torch.equal(value, self.model.backbone.state_dict()[key]))
        for key, original in self.before.items():
            if not key.startswith("backbone."):
                self.assertTrue(torch.equal(original, self.model.state_dict()[key]), key)
        self.assertEqual(provenance["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(provenance["bytes"], self.path.stat().st_size)
        self.assertEqual(provenance["tensor_count"], len(self.payload["backbone_state"]))
        self.assertEqual(provenance["source_provenance"], self.payload["provenance"])
        json.dumps(provenance, allow_nan=False)

    def test_missing_or_extra_tensor_keys_fail_before_any_weight_changes(self):
        import torch
        from holospex_ml.model import load_backbone_checkpoint

        complete = dict(self.payload["backbone_state"])
        for state in ({key: value for key, value in complete.items() if key != "0.weight"},
                      {**complete, "classifier.weight": torch.zeros(1)}):
            with self.subTest(keys=list(state)):
                self.payload["backbone_state"] = state
                self.write()
                with self.assertRaisesRegex(ValueError, "keys"):
                    load_backbone_checkpoint(self.model, self.path, architecture="deeplabv3_resnet50")
                self.assert_unchanged()

    def test_bad_shape_dtype_or_nonfinite_value_fails_without_partial_initialization(self):
        import torch
        from holospex_ml.model import load_backbone_checkpoint

        complete = dict(self.payload["backbone_state"])
        for bad in (torch.zeros(5), torch.zeros(4, dtype=torch.float64), torch.full((4,), float("nan"))):
            self.payload["backbone_state"] = {**complete, "1.running_var": bad}
            self.write()
            with self.assertRaises(ValueError):
                load_backbone_checkpoint(self.model, self.path, architecture="deeplabv3_resnet50")
            self.assert_unchanged()

    def test_wrong_architecture_format_and_non_json_provenance_are_rejected(self):
        import torch
        from holospex_ml.model import load_backbone_checkpoint

        self.write()
        with patch("holospex_ml.model.torch.load") as loader, self.assertRaisesRegex(ValueError, "only"):
            load_backbone_checkpoint(self.model, self.path, architecture="deeplabv3_mobilenet_v3_large")
        loader.assert_not_called()
        valid = dict(self.payload)
        for changes in ({"format_version": True}, {"architecture": "resnet18"}, {"provenance": {}},
                        {"provenance": {"value": float("nan")}}, {"provenance": {"value": torch.zeros(1)}}):
            self.payload = {**valid, **changes}
            self.write()
            with self.assertRaises(ValueError):
                load_backbone_checkpoint(self.model, self.path, architecture="deeplabv3_resnet50")
            self.assert_unchanged()

    def test_checkpoint_changed_during_read_is_rejected_before_mutation(self):
        import torch
        from holospex_ml.model import load_backbone_checkpoint

        self.write()
        original_load = torch.load

        def load_then_change(*args, **kwargs):
            payload = original_load(*args, **kwargs)
            with self.path.open("ab") as output:
                output.write(b"changed")
            return payload

        with patch("holospex_ml.model.torch.load", side_effect=load_then_change), \
                self.assertRaisesRegex(RuntimeError, "changed"):
            load_backbone_checkpoint(self.model, self.path, architecture="deeplabv3_resnet50")
        self.assert_unchanged()


if __name__ == "__main__":
    unittest.main()
