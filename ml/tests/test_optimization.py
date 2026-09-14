"""Small synthetic optimizer checks; never download weights or dataset pixels."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_TRAINING = all(importlib.util.find_spec(package) for package in ("torch", "torchvision", "numpy", "PIL"))
CLASSES = [
    {"index": 0, "sourceId": 0, "structureId": "background"},
    {"index": 1, "sourceId": 5, "structureId": "gallbladder"},
]


@unittest.skipUnless(HAS_TRAINING, "Training dependencies not installed")
class OptimizationTests(unittest.TestCase):
    def _model(self):
        import torch

        class TinySegmentation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Conv2d(3, 3, 1)
                self.classifier = torch.nn.Conv2d(3, 2, 1)

            def forward(self, pixels):
                return {"out": self.classifier(self.backbone(pixels))}

        return TinySegmentation()

    def _manifest(self, directory):
        import numpy as np
        from PIL import Image

        samples = []
        for case, split in ((1, "train"), (2, "train"), (3, "val"), (4, "test")):
            image, mask = directory / f"{case}.jpg", directory / f"{case}.png"
            Image.fromarray(np.full((2, 4, 3), 127, dtype=np.uint8)).save(image)
            Image.fromarray(np.array([[0, 0, 5, 255], [0, 5, 5, 255]], dtype=np.uint8)).save(mask)
            samples.append({"split": split, "videoId": case, "imagePath": str(image), "maskPath": str(mask)})
        return {"dataset": "synthetic-optimizer-test", "classes": copy.deepcopy(CLASSES),
                "ignoreIndex": 255, "ignoreSourceIds": [255], "samples": samples}

    def test_cosine_warmup_and_constant_rates_have_expected_endpoints(self):
        from holospex_ml.training import _learning_rate_factor

        self.assertEqual([_learning_rate_factor(step, 4, "none", 0) for step in range(5)], [1.0] * 5)
        for step, expected in enumerate((1.0, (1 + math.sqrt(0.5)) / 2, 0.5, (1 - math.sqrt(0.5)) / 2, 0.0)):
            self.assertAlmostEqual(_learning_rate_factor(step, 4, "cosine", 0), expected)
        self.assertEqual([_learning_rate_factor(step, 6, "cosine", 2) for step in (0, 1, 2)], [0.5, 1.0, 1.0])
        self.assertAlmostEqual(_learning_rate_factor(4, 6, "cosine", 2), 0.5)
        self.assertEqual(_learning_rate_factor(6, 6, "cosine", 2), 0.0)

    def test_adamw_updates_backbone_more_slowly_and_omits_frozen_tensors(self):
        import torch
        from holospex_ml.training import _optimizer_groups

        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(1, 1, bias=False)
        model.classifier = torch.nn.Linear(1, 1, bias=False)
        model.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        with torch.no_grad():
            model.backbone.weight.fill_(1.0)
            model.classifier.weight.fill_(1.0)
        groups = _optimizer_groups(model, 0.1, 0.1)
        selected = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(set(selected), {id(model.backbone.weight), id(model.classifier.weight)})
        optimizer = torch.optim.AdamW(groups, lr=0.1, weight_decay=0.01)
        for parameter in (model.backbone.weight, model.classifier.weight):
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertAlmostEqual(model.backbone.weight.item(), 1.0 - 0.01 * 1.01, places=6)
        self.assertAlmostEqual(model.classifier.weight.item(), 1.0 - 0.1 * 1.01, places=6)
        self.assertEqual(model.frozen.item(), 1.0)
        self.assertEqual(len(_optimizer_groups(model, 0.1, 1.0)), 1)
        with self.assertRaisesRegex(ValueError, "requires model.backbone"):
            _optimizer_groups(torch.nn.Linear(1, 1), 0.1, 0.1)

    def test_training_records_rates_actually_used_and_reproducible_config_hash(self):
        import torch
        from holospex_ml.training import _manifest_digest, train

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self._manifest(directory)
            with patch("holospex_ml.training.build_model", return_value=self._model()), \
                 patch("holospex_ml.training._evaluate_model", return_value={"foreground_macro_iou": 0.4}):
                report = train(manifest, directory / "run", epochs=2, batch_size=1, width=4, height=2,
                               pretrained=False, device="cpu", lr=0.01, lr_schedule="cosine",
                               backbone_lr_multiplier=0.1, weight_decay=0.02)
            self.assertEqual(report["epochs_completed"], 2)
            self.assertEqual(report["stop_reason"], "epochs_completed")
            first, second = [row["learning_rates"] for row in report["history"]]
            self.assertAlmostEqual(first["first_update"]["heads"], 0.01)
            self.assertAlmostEqual(first["last_update"]["heads"], 0.01 * (1 + math.sqrt(0.5)) / 2)
            self.assertAlmostEqual(second["first_update"]["heads"], 0.005)
            self.assertAlmostEqual(second["last_update"]["heads"], 0.01 * (1 - math.sqrt(0.5)) / 2)
            for row in (first, second):
                for rates in row.values():
                    self.assertAlmostEqual(rates["backbone"] / rates["heads"], 0.1)
            config = report["config"]
            self.assertEqual(config["weight_decay"], 0.02)
            self.assertEqual(config["config_sha256"], _manifest_digest({key: value for key, value in config.items() if key != "config_sha256"}))
            saved = torch.load(report["last_checkpoint"], map_location="cpu", weights_only=True)
            self.assertEqual(saved["training"]["config_sha256"], config["config_sha256"])
            self.assertEqual(json.loads((directory / "run" / "config.json").read_text()), config)

    def test_duration_stop_preserves_actual_completed_epoch_and_both_checkpoints(self):
        import torch
        from holospex_ml.training import train

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self._manifest(directory)
            with patch("holospex_ml.training.build_model", return_value=self._model()), \
                 patch("holospex_ml.training._evaluate_model", return_value={"foreground_macro_iou": 0.4}), \
                 patch("holospex_ml.training.time.monotonic", side_effect=[0.0, 0.0, 2.0, 3.0, 4.0]):
                report = train(manifest, directory / "run", epochs=4, batch_size=1, width=4, height=2,
                               pretrained=False, device="cpu", max_duration_seconds=1.0)
            self.assertEqual(report["epochs_completed"], 1)
            self.assertEqual(report["config"]["epochs"], 4)
            self.assertEqual(report["stop_reason"], "max_duration_seconds")
            self.assertEqual(report["duration_seconds"], 4.0)
            self.assertEqual(len(json.loads((directory / "run" / "history.json").read_text())), 1)
            for name in ("checkpoint", "last_checkpoint"):
                checkpoint = torch.load(report[name], map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["epochs_trained"], 1)

    def test_safe_warm_start_reuses_weights_and_records_source_not_new_pretraining(self):
        import torch
        from holospex_ml.model import ARCHITECTURE
        from holospex_ml.training import train

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self._manifest(directory)
            initial_model = self._model()
            with patch("holospex_ml.training.build_model", return_value=initial_model), \
                 patch("holospex_ml.training._evaluate_model", return_value={"foreground_macro_iou": 0.4}):
                source = train(manifest, directory / "source", epochs=1, width=4, height=2, pretrained=False, device="cpu")
            source_path = Path(source["checkpoint"])
            loaded_model = self._model()
            before = {}
            def remember_loaded(model):
                if not before:
                    before.update({key: value.detach().clone() for key, value in model.state_dict().items()})
            # load_checkpoint uses its real weights-only reader and strict state
            # loading, with only the heavyweight architecture replaced locally.
            with patch("holospex_ml.model.build_model", return_value=loaded_model) as safe_builder, \
                 patch("holospex_ml.training.build_model") as fresh_builder, \
                 patch("holospex_ml.training.freeze_batchnorm", side_effect=remember_loaded), \
                 patch("holospex_ml.training._evaluate_model", return_value={"foreground_macro_iou": 0.5}):
                report = train(manifest, directory / "warm", epochs=1, width=4, height=2,
                               device="cpu", initial_checkpoint=source_path)
            fresh_builder.assert_not_called()
            safe_builder.assert_called_once_with(num_classes=2, pretrained=False, architecture=ARCHITECTURE)
            for name, expected in initial_model.state_dict().items():
                torch.testing.assert_close(before[name], expected)
            self.assertFalse(report["config"]["pretrained"])
            self.assertEqual(report["config"]["initialization"], "checkpoint")
            provenance = report["config"]["initial_checkpoint"]
            self.assertEqual(provenance["checkpoint_sha256"], hashlib.sha256(source_path.read_bytes()).hexdigest())
            self.assertEqual(provenance["training_config_sha256"], source["config"]["config_sha256"])
            self.assertEqual(provenance["train_video_ids"], ["1", "2"])
            self.assertEqual(provenance["val_video_ids"], ["3"])

    def test_warm_start_rejects_changed_cases_class_map_and_ignore_policy(self):
        from holospex_ml.model import ARCHITECTURE
        from holospex_ml.training import _check_initial_checkpoint

        manifest = {"classes": CLASSES, "ignoreIndex": 255, "ignoreSourceIds": [255]}
        checkpoint = {"architecture": ARCHITECTURE, "classes": CLASSES, "ignore_index": 255,
                      "ignore_source_ids": [255], "training": {"train_video_ids": ["1"], "val_video_ids": ["2"]}}
        _check_initial_checkpoint(checkpoint, manifest, ARCHITECTURE, [{"videoId": 1}], [{"videoId": 2}])
        for training, validation in (([2], [1]), ([1, 2], [3]), ([3], [2]), ([1], [3])):
            with self.subTest(training=training, validation=validation), self.assertRaisesRegex(ValueError, "same selected"):
                _check_initial_checkpoint(checkpoint, manifest, ARCHITECTURE,
                                          [{"videoId": case} for case in training], [{"videoId": case} for case in validation])
        changes = (("architecture", "unknown", "architecture"), ("classes", list(reversed(CLASSES)), "class maps"),
                   ("ignore_source_ids", [], "policies differ"), ("training", {}, "case provenance"))
        for key, value, message in changes:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                _check_initial_checkpoint({**checkpoint, key: value}, manifest, ARCHITECTURE, [{"videoId": 1}], [{"videoId": 2}])

    def test_invalid_options_fail_before_constructing_model(self):
        from holospex_ml.training import train

        manifest = {"classes": CLASSES, "samples": []}
        options = ({"lr_schedule": "polynomial"}, {"warmup_epochs": 3}, {"warmup_epochs": True},
                   {"weight_decay": -1.0}, {"weight_decay": float("nan")},
                   {"backbone_lr_multiplier": 0.0}, {"max_duration_seconds": 0.0})
        for settings in options:
            with self.subTest(settings=settings), patch("holospex_ml.training.build_model") as builder:
                with self.assertRaises(ValueError):
                    train(manifest, Path("unused"), **settings)
                builder.assert_not_called()

    def test_cli_passes_controls_without_downloads(self):
        from holospex_ml.cli import main

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = directory / "manifest.json"
            manifest.write_text("{}")
            with patch("holospex_ml.training.train", return_value={}) as runner:
                result = main(["train", "--manifest", str(manifest), "--output-dir", str(directory / "out"),
                               "--lr-schedule", "cosine", "--warmup-epochs", "1", "--weight-decay", "0.02",
                               "--backbone-lr-multiplier", "0.1", "--max-duration-seconds", "500",
                               "--initial-checkpoint", str(directory / "source.pt"),
                               "--architecture", "deeplabv3plus_mobilenet_v3_large"])
            self.assertEqual(result, 0)
            settings = runner.call_args.kwargs
            self.assertEqual(settings["lr_schedule"], "cosine")
            self.assertEqual(settings["warmup_epochs"], 1)
            self.assertEqual(settings["weight_decay"], 0.02)
            self.assertEqual(settings["backbone_lr_multiplier"], 0.1)
            self.assertEqual(settings["max_duration_seconds"], 500)
            self.assertEqual(settings["initial_checkpoint"], directory / "source.pt")


if __name__ == "__main__":
    unittest.main()
