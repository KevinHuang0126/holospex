"""Architecture compatibility and train-only experiment invariants; no downloads."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


CLASSES = [
    {"index": 0, "sourceId": 0, "structureId": "background"},
    {"index": 1, "sourceId": 5, "structureId": "gallbladder"},
    {"index": 2, "sourceId": 4, "structureId": "cystic_duct"},
]


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"), "Training dependencies not installed")
class CloudExperimentTests(unittest.TestCase):
    def test_both_backbones_produce_requested_main_and_aux_channels_without_download(self):
        import torch
        from holospex_ml.model import ARCHITECTURE, build_model

        for architecture in (ARCHITECTURE, "deeplabv3_resnet50"):
            with self.subTest(architecture=architecture):
                model = build_model(3, pretrained=False, architecture=architecture).eval()
                with torch.inference_mode():
                    output = model(torch.zeros(1, 3, 32, 64))
                self.assertEqual(tuple(output["out"].shape), (1, 3, 32, 64))
                self.assertEqual(tuple(output["aux"].shape), (1, 3, 32, 64))

    def test_generic_pretrained_constructor_loads_original_heads_before_replacement(self):
        import torch
        from holospex_ml.model import DeepLabV3_ResNet50_Weights, build_model

        class GenericHeads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Sequential(torch.nn.Conv2d(8, 21, 1))
                self.aux_classifier = torch.nn.Sequential(torch.nn.Conv2d(4, 21, 1))

        with patch("holospex_ml.model.deeplabv3_resnet50", return_value=GenericHeads()) as constructor:
            model = build_model(7, pretrained=True, architecture="deeplabv3_resnet50")
        constructor.assert_called_once_with(weights=DeepLabV3_ResNet50_Weights.DEFAULT, weights_backbone=None, aux_loss=True)
        self.assertEqual((model.classifier[-1].in_channels, model.classifier[-1].out_channels), (8, 7))
        self.assertEqual((model.aux_classifier[-1].in_channels, model.aux_classifier[-1].out_channels), (4, 7))

    def test_old_and_new_checkpoint_architectures_dispatch_and_restore_exact_state(self):
        import torch
        from holospex_ml.model import ARCHITECTURE, load_checkpoint, model_id_for_architecture

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.pt"
            for architecture in (ARCHITECTURE, "deeplabv3_resnet50"):
                original = torch.nn.Conv2d(3, 3, 1)
                checkpoint = {"format_version": 1, "architecture": architecture,
                              "model_id": model_id_for_architecture(architecture), "classes": CLASSES,
                              "input_size": {"width": 64, "height": 32}, "model_state": original.state_dict()}
                torch.save(checkpoint, path)
                with patch("holospex_ml.model.build_model", return_value=torch.nn.Conv2d(3, 3, 1)) as builder:
                    loaded, metadata = load_checkpoint(path, "cpu")
                builder.assert_called_once_with(num_classes=3, pretrained=False, architecture=architecture)
                self.assertFalse(loaded.training)
                self.assertEqual(metadata["model_id"], checkpoint["model_id"])
                torch.testing.assert_close(loaded.weight, original.weight)
            checkpoint["architecture"] = "unknown_backbone"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "Unsupported architecture"):
                load_checkpoint(path, "cpu")

    def test_augmentation_flips_image_and_ignore_mask_together_without_label_changes(self):
        from PIL import Image
        from holospex_ml.training import _augment_pair

        pixels = np.array([[[20, 30, 40], [80, 90, 100], [140, 150, 160]],
                           [[30, 40, 50], [90, 100, 110], [150, 160, 170]]], dtype=np.uint8)
        target = np.array([[0, 255, 1], [2, 2, 255]], dtype=np.int32)
        with patch("holospex_ml.training.random.random", return_value=0.0), patch("holospex_ml.training.random.uniform", return_value=1.0):
            image, mask = _augment_pair(Image.fromarray(pixels), Image.fromarray(target))
        np.testing.assert_array_equal(np.array(image), pixels[:, ::-1])
        np.testing.assert_array_equal(np.array(mask), target[:, ::-1])
        self.assertEqual(np.count_nonzero(np.array(mask) == 255), 2)
        with patch("holospex_ml.training.random.random", return_value=1.0), patch("holospex_ml.training.random.uniform", side_effect=[1.1, 0.9]):
            image, mask = _augment_pair(Image.fromarray(pixels), Image.fromarray(target))
        self.assertFalse(np.array_equal(np.array(image), pixels))
        np.testing.assert_array_equal(np.array(mask), target)

    def test_augmentation_rejects_validation_test_and_unmarked_inputs(self):
        from holospex_ml.training import SegmentationDataset

        for split in ("val", "test", None):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "training samples"):
                SegmentationDataset([{"split": split}], CLASSES, 6, 4, augmentation="mild")

    def test_case_sampler_equalizes_case_probability_and_remains_seeded(self):
        import torch
        from torch.utils.data import WeightedRandomSampler
        from holospex_ml.training import _case_balanced_weights

        samples = [{"split": "train", "videoId": video} for video in (7, "7", 7, 9)]
        weights = _case_balanced_weights(samples)
        self.assertAlmostEqual(sum(weights[:3]), weights[3])
        draws = [list(WeightedRandomSampler(weights, 10000, replacement=True,
                                           generator=torch.Generator().manual_seed(42))) for _ in range(2)]
        self.assertEqual(draws[0], draws[1])
        self.assertAlmostEqual(sum(index == 3 for index in draws[0]) / len(draws[0]), 0.5, delta=0.02)
        for invalid in ([], samples + [{"split": "val", "videoId": 10}]):
            with self.assertRaisesRegex(ValueError, "training samples"):
                _case_balanced_weights(invalid)

    def test_training_records_experiments_without_augmenting_validation_or_class_counts(self):
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler
        from holospex_ml.training import _augment_pair, train

        class TinySegmentation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Conv2d(3, 3, 1)

            def forward(self, pixels):
                return {"out": self.head(pixels)}

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            samples = []
            target = np.array([[0, 255, 5, 5, 4, 4]] * 4, dtype=np.uint8)
            for index, (video, split) in enumerate([(1, "train"), (1, "train"), (2, "train"), (10, "val"), (11, "test")]):
                image_path, mask_path = directory / f"{index}.jpg", directory / f"{index}.png"
                Image.fromarray(np.full((4, 6, 3), 127, dtype=np.uint8)).save(image_path)
                Image.fromarray(target).save(mask_path)
                samples.append({"split": split, "videoId": video, "imagePath": str(image_path), "maskPath": str(mask_path)})
            manifest = {"classes": CLASSES, "samples": samples, "ignoreIndex": 255, "ignoreSourceIds": [255]}
            loaders = []

            def capture_loader(*args, **kwargs):
                loader = DataLoader(*args, **kwargs)
                loaders.append(loader)
                return loader

            with patch("holospex_ml.training.build_model", return_value=TinySegmentation()) as builder, \
                 patch("holospex_ml.training._augment_pair", wraps=_augment_pair) as augmenter, \
                 patch("holospex_ml.training.DataLoader", side_effect=capture_loader):
                report = train(manifest, directory / "run", epochs=1, batch_size=1, width=6, height=4,
                               device="cpu", pretrained=False, class_weighting="balanced",
                               architecture="deeplabv3_resnet50", augmentation="mild", sampling="case_balanced")
            builder.assert_called_once_with(3, pretrained=False, architecture="deeplabv3_resnet50")
            self.assertEqual(augmenter.call_count, 3)  # Each train draw once; never weights or validation.
            self.assertIsInstance(loaders[0].sampler, WeightedRandomSampler)
            self.assertIsInstance(loaders[1].sampler, SequentialSampler)
            self.assertEqual(loaders[1].dataset.augmentation, "none")
            config = report["config"]
            self.assertEqual(config["class_weighting_details"]["counts"], [12, 24, 24])
            self.assertEqual(config["class_weighting_details"]["ignored_pixels"], 12)
            self.assertEqual(config["sampling_details"]["video_frame_counts"], {"1": 2, "2": 1})
            self.assertEqual(config["sampling_details"]["draws_per_epoch"], 3)
            self.assertEqual(config["val_samples"], 1)
            checkpoint = torch.load(report["checkpoint"], weights_only=True)
            self.assertEqual(checkpoint["architecture"], "deeplabv3_resnet50")
            self.assertEqual(checkpoint["model_id"], "holospex-deeplabv3-resnet50")
            self.assertEqual(checkpoint["training"]["augmentation"], "mild")
            self.assertEqual(checkpoint["training"]["sampling"], "case_balanced")


if __name__ == "__main__":
    unittest.main()
