"""Backbone initialization integration without downloading external weights."""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_vertex_bootstrap import bootstrap
from test_vertex_entry import entry, FakeBucket

HAS_TRAINING = all(importlib.util.find_spec(package) for package in ("torch", "torchvision", "numpy", "PIL"))
CLASSES = [{"index": 0, "sourceId": 0, "structureId": "background"},
           {"index": 1, "sourceId": 1, "structureId": "gallbladder"}]
ARCHITECTURE = "deeplabv3_resnet50"


@unittest.skipUnless(HAS_TRAINING, "Training dependencies not installed")
class BackboneTrainingTests(unittest.TestCase):
    def test_backbone_overlay_precedes_optimizer_and_preserves_context_initialization(self):
        import numpy as np
        from PIL import Image
        import torch
        from holospex_ml.model import load_backbone_checkpoint
        from holospex_ml.training import train

        class TinySegmentation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Conv2d(3, 3, 1, bias=False)
                self.classifier = torch.nn.Conv2d(3, 2, 1, bias=False)
                with torch.no_grad():
                    self.backbone.weight.fill_(1.0)
                    self.classifier.weight.fill_(.5)

            def forward(self, image):
                return {"out": self.classifier(self.backbone(image))}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "backbone.pt"
            model = TinySegmentation()
            backbone_state = {key: torch.full_like(value, 3.0) for key, value in model.backbone.state_dict().items()}
            torch.save({"format_version": 1, "artifact_type": "holospex_pretrained_backbone", "architecture": "resnet50",
                        "backbone_state": backbone_state, "provenance": {"dataset": "synthetic-test-only"}}, source)
            metadata = {"architecture": ARCHITECTURE, "path": str(source.resolve()),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bytes": source.stat().st_size,
                        "tensor_count": len(backbone_state), "source_provenance": {"dataset": "synthetic-test-only"}}
            samples = []
            for case, split in ((1, "train"), (2, "val")):
                image_path, mask_path = root / f"{case}.jpg", root / f"{case}.png"
                Image.fromarray(np.full((2, 4, 3), 127, dtype=np.uint8)).save(image_path)
                Image.fromarray(np.array([[0, 1, 1, 255]] * 2, dtype=np.uint8)).save(mask_path)
                samples.append({"split": split, "videoId": case, "imagePath": str(image_path), "maskPath": str(mask_path)})
            manifest = {"classes": CLASSES, "samples": samples, "ignoreIndex": 255, "ignoreSourceIds": [255]}
            events = []
            def apply(model, path, *, architecture):
                self.assertEqual(architecture, ARCHITECTURE)
                self.assertTrue(torch.equal(model.backbone.weight, torch.ones_like(model.backbone.weight)))
                result = load_backbone_checkpoint(model, path, architecture=architecture)
                events.append("backbone_loaded")
                return result
            optimizer_constructor = torch.optim.AdamW
            def optimizer(*args, **kwargs):
                self.assertEqual(events, ["backbone_loaded"])
                self.assertTrue(torch.equal(model.backbone.weight, torch.full_like(model.backbone.weight, 3.0)))
                self.assertTrue(torch.equal(model.classifier.weight, torch.full_like(model.classifier.weight, .5)))
                events.append("optimizer_created")
                return optimizer_constructor(*args, **kwargs)
            with patch("holospex_ml.training.build_model", return_value=model) as builder, \
                 patch("holospex_ml.training.load_backbone_checkpoint", side_effect=apply) as overlay, \
                 patch("holospex_ml.training.torch.optim.AdamW", side_effect=optimizer):
                report = train(manifest, root / "run", epochs=1, width=4, height=2, device="cpu",
                               architecture=ARCHITECTURE, pretrained=True, backbone_checkpoint=source)
            builder.assert_called_once_with(2, pretrained=True, architecture=ARCHITECTURE)
            overlay.assert_called_once_with(model, source, architecture=ARCHITECTURE)
            self.assertEqual(events, ["backbone_loaded", "optimizer_created"])
            config = report["config"]
            self.assertEqual(config["initialization"], "COCO_WITH_VOC_LABELS_V1+backbone_checkpoint")
            self.assertTrue(config["pretrained"])
            self.assertIsNone(config["initial_checkpoint"])
            self.assertEqual(config["backbone_checkpoint"], metadata)
            self.assertEqual(config["initialization_recipe"]["retained_weights"], "COCO_ASPP_context_and_auxiliary_feature_layers")
            checkpoint = torch.load(report["checkpoint"], weights_only=True)
            self.assertEqual(checkpoint["backbone_checkpoint"], metadata)
            self.assertEqual(checkpoint["initialization_recipe"], config["initialization_recipe"])
            self.assertEqual(checkpoint["training"]["backbone_checkpoint"]["sha256"], metadata["sha256"])
            self.assertFalse(torch.equal(model.classifier.weight, torch.full_like(model.classifier.weight, .5)),
                             "A real training update must follow initialization")

    def test_invalid_initialization_combinations_fail_before_model_or_download(self):
        from holospex_ml.training import train

        cases = [({"initial_checkpoint": Path("initial.pt")}, "mutually exclusive"),
                 ({"architecture": "deeplabv3_mobilenet_v3_large"}, "requires deeplabv3_resnet50"),
                 ({"pretrained": False}, "pretrained=True"), ({}, "existing checkpoint")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for overrides, message in cases:
                options = {"architecture": ARCHITECTURE, "pretrained": True, "backbone_checkpoint": root / "missing.pt", **overrides}
                with self.subTest(options=options), patch("holospex_ml.training.build_model") as builder:
                    with self.assertRaisesRegex(ValueError, message):
                        train({"classes": CLASSES, "samples": []}, root / "out", **options)
                    builder.assert_not_called()
                    self.assertFalse((root / "out").exists())

    def test_cli_passes_backbone_checkpoint_with_explicit_architecture(self):
        from holospex_ml.cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            with patch("holospex_ml.training.train", return_value={}) as runner:
                self.assertEqual(main(["train", "--manifest", str(manifest), "--output-dir", str(root / "out"),
                                       "--architecture", ARCHITECTURE, "--backbone-checkpoint", str(root / "backbone.pt")]), 0)
            self.assertEqual(runner.call_args.kwargs["backbone_checkpoint"], root / "backbone.pt")
            self.assertTrue(runner.call_args.kwargs["pretrained"])


class BackboneCloudTests(unittest.TestCase):
    def environment(self):
        return {"HOLOSPEX_SOURCE_URI": "gs://bucket/source.tar.gz", "HOLOSPEX_SOURCE_SHA256": "0" * 64,
                "HOLOSPEX_DATA_URI": "gs://bucket/data.tar.gz", "HOLOSPEX_DATA_SHA256": "1" * 64,
                "HOLOSPEX_OUTPUT_URI": "gs://bucket/runs/test", "HOLOSPEX_RUN_NAME": "test", "HOLOSPEX_EPOCHS": "4",
                "HOLOSPEX_ARCHITECTURE": ARCHITECTURE, "HOLOSPEX_BACKBONE_CHECKPOINT_URI": "gs://bucket/backbone.pt",
                "HOLOSPEX_BACKBONE_CHECKPOINT_SHA256": "2" * 64}

    def test_uri_sha_architecture_and_exclusivity_are_checked_before_runtime(self):
        changes = [{"HOLOSPEX_BACKBONE_CHECKPOINT_URI": ""}, {"HOLOSPEX_BACKBONE_CHECKPOINT_SHA256": ""},
                   {"HOLOSPEX_BACKBONE_CHECKPOINT_SHA256": "invalid"}, {"HOLOSPEX_BACKBONE_CHECKPOINT_URI": "https://bucket/backbone.pt"},
                   {"HOLOSPEX_ARCHITECTURE": "deeplabv3_mobilenet_v3_large"},
                   {"HOLOSPEX_INITIAL_CHECKPOINT_URI": "gs://bucket/initial.pt", "HOLOSPEX_INITIAL_CHECKPOINT_SHA256": "3" * 64}]
        for change in changes:
            with self.subTest(change=change), patch.dict(bootstrap.os.environ, {**self.environment(), **change}, clear=True), \
                 patch.object(bootstrap, "verify_runtime") as runtime, patch.object(bootstrap.subprocess, "run") as process:
                with self.assertRaises(ValueError):
                    bootstrap.main()
                runtime.assert_not_called()
                process.assert_not_called()

    def test_worker_local_checkpoint_is_forwarded_only_to_training(self):
        self.assertTrue(bootstrap.backbone_checkpoint_enabled(self.environment()))
        command = bootstrap.entry_command(self.environment(), Path("/work"))
        self.assertEqual(command[command.index("--backbone-checkpoint") + 1], "/work/backbone.pt")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "backbone.pt"
            path.write_bytes(b"fixture")
            args = argparse.Namespace(data_root=root, output_uri="gs://bucket/runs/test", run_name="test", epochs=4,
                                      width=672, height=384, architecture=ARCHITECTURE, seed=42,
                                      backbone_checkpoint=path, work_dir=root / "out")
            entry.validate_optimization_args(args)
            commands = entry.build_commands(args, root)
            train = next(command for phase, command, _ in commands if phase == "train")
            evaluation = next(command for phase, command, _ in commands if phase == "evaluate")
            self.assertEqual(train[train.index("--backbone-checkpoint") + 1], str(path))
            self.assertNotIn("--backbone-checkpoint", evaluation)
            for overrides in ({"initial_checkpoint": path}, {"architecture": None}, {"backbone_checkpoint": root / "missing"}):
                bad = argparse.Namespace(**{**vars(args), **overrides})
                bucket = FakeBucket()
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    entry.execute(bad, bucket)
                self.assertFalse(bucket.objects)
                self.assertFalse(args.work_dir.exists())


if __name__ == "__main__":
    unittest.main()
