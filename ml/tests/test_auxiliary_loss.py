"""Auxiliary-head ablations preserve model shape and measure real gradients."""

import argparse
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_vertex_bootstrap import bootstrap
from test_vertex_entry import entry, FakeBucket

HAS_TRAINING = all(importlib.util.find_spec(package) for package in ("torch", "torchvision", "numpy", "PIL"))
CLASSES = [{"index": i, "sourceId": i, "structureId": name}
           for i, name in enumerate(("background", "gallbladder", "cystic_duct"))]


@unittest.skipUnless(HAS_TRAINING, "Training dependencies not installed")
class AuxiliaryLossTests(unittest.TestCase):
    def model(self):
        import torch

        class TinyHeads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.main = torch.nn.Parameter(torch.tensor([.2, -.1, .3]))
                self.auxiliary = torch.nn.Parameter(torch.tensor([-.4, .7, .2]))

            def forward(self, image):
                shape = (image.shape[0], 3, *image.shape[-2:])
                return {"out": self.main.view(1, 3, 1, 1).expand(shape),
                        "aux": self.auxiliary.view(1, 3, 1, 1).expand(shape)}
        return TinyHeads()

    def manifest(self, root):
        import numpy as np
        from PIL import Image

        samples = []
        for case, split in ((1, "train"), (2, "val")):
            image_path, mask_path = root / f"{case}.jpg", root / f"{case}.png"
            Image.fromarray(np.full((2, 4, 3), 127, dtype=np.uint8)).save(image_path)
            Image.fromarray(np.array([[0, 1, 2, 255], [1, 1, 2, 255]], dtype=np.uint8)).save(mask_path)
            samples.append({"split": split, "videoId": case, "imagePath": str(image_path), "maskPath": str(mask_path)})
        return {"classes": CLASSES, "samples": samples, "ignoreIndex": 255, "ignoreSourceIds": [255]}

    def test_zero_auxiliary_weight_matches_main_gradient_and_prevents_auxiliary_adamw_update(self):
        import torch
        from holospex_ml.losses import foreground_generalized_dice_loss, foreground_lovasz_softmax_loss
        from holospex_ml.training import train

        target = torch.tensor([[[0, 1, 2, 255], [1, 1, 2, 255]]])
        for loss_mode in ("ce", "ce_generalized_dice", "ce_lovasz"):
            with self.subTest(loss=loss_mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model, reference = self.model(), self.model()
                before = copy.deepcopy(model.state_dict())
                outputs = reference(torch.zeros(1, 3, 2, 4))
                expected = torch.nn.functional.cross_entropy(outputs["out"], target, ignore_index=255)
                if loss_mode == "ce_generalized_dice":
                    expected = expected + foreground_generalized_dice_loss(outputs["out"], target)
                if loss_mode == "ce_lovasz":
                    expected = expected + .25 * foreground_lovasz_softmax_loss(outputs["out"], target)
                expected.backward()
                with patch("holospex_ml.training.build_model", return_value=model):
                    report = train(self.manifest(root), root / "run", epochs=1, width=4, height=2,
                                   device="cpu", pretrained=False, loss=loss_mode, auxiliary_loss_weight=0,
                                   weight_decay=.1)
                self.assertIsNone(model.auxiliary.grad)
                torch.testing.assert_close(model.auxiliary, before["auxiliary"], atol=0, rtol=0)
                torch.testing.assert_close(model.main.grad, reference.main.grad)
                self.assertGreater(model.main.grad.abs().sum().item(), 0)
                self.assertFalse(torch.equal(model.main, before["main"]))
                self.assertAlmostEqual(report["history"][0]["train_loss"], expected.item(), places=6)
                self.assertEqual(report["config"]["auxiliary_loss_weight"], 0)
                self.assertFalse(report["config"]["loss_details"]["auxiliary_loss_enabled"])
                components = report["history"][0]["train_loss_components"]
                self.assertEqual(components["auxiliary_head_weight"], 0)
                self.assertIsNone(components["per_head"]["ce_aux"])
                self.assertIsNone(components["per_head"]["dice_aux"])
                checkpoint = torch.load(report["checkpoint"], weights_only=True)
                self.assertEqual(checkpoint["auxiliary_loss_weight"], 0)
                self.assertEqual(set(checkpoint["model_state"]), set(before))
                torch.testing.assert_close(checkpoint["model_state"]["auxiliary"], before["auxiliary"], atol=0, rtol=0)

    def test_default_and_nondefault_auxiliary_coefficients_match_loss_and_gradient(self):
        import torch
        from holospex_ml.training import train

        target = torch.tensor([[[0, 1, 2, 255], [1, 1, 2, 255]]])
        for coefficient in (.4, .7):
            with self.subTest(coefficient=coefficient), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model, reference = self.model(), self.model()
                output = reference(torch.zeros(1, 3, 2, 4))
                expected = (torch.nn.functional.cross_entropy(output["out"], target, ignore_index=255)
                            + coefficient * torch.nn.functional.cross_entropy(output["aux"], target, ignore_index=255))
                expected.backward()
                options = {} if coefficient == .4 else {"auxiliary_loss_weight": coefficient}
                with patch("holospex_ml.training.build_model", return_value=model):
                    report = train(self.manifest(root), root / "run", epochs=1, width=4, height=2,
                                   device="cpu", pretrained=False, **options)
                self.assertAlmostEqual(report["history"][0]["train_loss"], expected.item(), places=6)
                torch.testing.assert_close(model.main.grad, reference.main.grad)
                torch.testing.assert_close(model.auxiliary.grad, reference.auxiliary.grad)
                self.assertGreater(model.auxiliary.grad.abs().sum().item(), 0)
                self.assertEqual(report["config"]["auxiliary_loss_weight"], coefficient)
                self.assertEqual(report["config"]["loss_details"]["batch_objective"], f"CE_main + {coefficient}*CE_aux")

    def test_invalid_coefficients_fail_before_model_initialization(self):
        from holospex_ml.training import train

        for coefficient in (-1, 10.1, float("nan"), float("inf"), True):
            with self.subTest(coefficient=coefficient), patch("holospex_ml.training.build_model") as builder:
                with self.assertRaisesRegex(ValueError, "auxiliary_loss_weight"):
                    train({"classes": CLASSES, "samples": []}, Path("unused"), auxiliary_loss_weight=coefficient)
                builder.assert_not_called()

    def test_main_only_warm_start_preserves_existing_auxiliary_checkpoint_parameters(self):
        import torch
        from holospex_ml.training import train

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            with patch("holospex_ml.training.build_model", return_value=self.model()):
                source = train(manifest, root / "source", epochs=1, width=4, height=2,
                               device="cpu", pretrained=False)
            checkpoint = torch.load(source["checkpoint"], weights_only=True)
            warmed_model = self.model()
            with patch("holospex_ml.model.build_model", return_value=warmed_model), \
                 patch("holospex_ml.training.build_model") as fresh_builder:
                report = train(manifest, root / "warm", epochs=1, width=4, height=2, device="cpu",
                               initial_checkpoint=Path(source["checkpoint"]), auxiliary_loss_weight=0, weight_decay=.1)
            fresh_builder.assert_not_called()
            self.assertEqual(report["config"]["initialization"], "checkpoint")
            self.assertEqual(report["config"]["auxiliary_loss_weight"], 0)
            self.assertIsNone(warmed_model.auxiliary.grad)
            torch.testing.assert_close(warmed_model.auxiliary, checkpoint["model_state"]["auxiliary"], atol=0, rtol=0)

    def test_training_cli_propagates_zero_without_changing_default(self):
        from holospex_ml.cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            arguments = ["train", "--manifest", str(manifest), "--output-dir", str(root / "out")]
            for additional, expected in (([], .4), (["--auxiliary-loss-weight", "0"], 0)):
                with patch("holospex_ml.training.train", return_value={}) as runner:
                    self.assertEqual(main(arguments + additional), 0)
                self.assertEqual(runner.call_args.kwargs["auxiliary_loss_weight"], expected)


class AuxiliaryCloudTests(unittest.TestCase):
    def environment(self):
        return {"HOLOSPEX_SOURCE_URI": "gs://bucket/source.tar.gz", "HOLOSPEX_SOURCE_SHA256": "0" * 64,
                "HOLOSPEX_DATA_URI": "gs://bucket/data.tar.gz", "HOLOSPEX_DATA_SHA256": "1" * 64,
                "HOLOSPEX_OUTPUT_URI": "gs://bucket/runs/test", "HOLOSPEX_RUN_NAME": "test", "HOLOSPEX_EPOCHS": "4"}

    def test_zero_and_nondefault_coefficients_propagate_through_both_cloud_stages(self):
        for coefficient in (0, .7):
            environment = {**self.environment(), "HOLOSPEX_AUXILIARY_LOSS_WEIGHT": str(coefficient)}
            bootstrap.validate_optimization_environment(environment)
            command = bootstrap.entry_command(environment, Path("/work"))
            self.assertEqual(command[command.index("--auxiliary-loss-weight") + 1], str(coefficient))
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = argparse.Namespace(data_root=root, epochs=4, seed=42, width=672, height=384,
                                          auxiliary_loss_weight=coefficient)
                entry.validate_optimization_args(args)
                for phase, command, _ in entry.build_commands(args, root):
                    if phase == "train":
                        self.assertEqual(command[command.index("--auxiliary-loss-weight") + 1], str(coefficient))
                    else:
                        self.assertNotIn("--auxiliary-loss-weight", command)
        self.assertNotIn("--auxiliary-loss-weight", bootstrap.entry_command(self.environment(), Path("/work")))

    def test_invalid_cloud_coefficients_fail_before_runtime_or_attempt_creation(self):
        for coefficient in (-1, 10.1, float("nan"), float("inf")):
            environment = {**self.environment(), "HOLOSPEX_AUXILIARY_LOSS_WEIGHT": str(coefficient)}
            with self.subTest(coefficient=coefficient), patch.dict(bootstrap.os.environ, environment, clear=True), \
                 patch.object(bootstrap, "verify_runtime") as runtime:
                with self.assertRaisesRegex(ValueError, "AUXILIARY_LOSS_WEIGHT"):
                    bootstrap.main()
                runtime.assert_not_called()
            with tempfile.TemporaryDirectory() as temporary:
                root, bucket = Path(temporary), FakeBucket()
                args = argparse.Namespace(output_uri="gs://bucket/runs/test", epochs=4,
                                          auxiliary_loss_weight=coefficient, work_dir=root / "out")
                with self.assertRaisesRegex(ValueError, "auxiliary_loss_weight"):
                    entry.execute(args, bucket)
                self.assertFalse(bucket.objects)
                self.assertFalse(args.work_dir.exists())


if __name__ == "__main__":
    unittest.main()
