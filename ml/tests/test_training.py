import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from holospex_ml.metrics import confusion_matrix, metrics_from_confusion


CLASSES = [
    {"index": 0, "sourceId": 0, "structureId": "background"},
    {"index": 1, "sourceId": 5, "structureId": "gallbladder"},
    {"index": 2, "sourceId": 4, "structureId": "cystic_duct"},
]


class SegmentationMetricsTests(unittest.TestCase):
    def test_ignored_pixels_are_neither_background_nor_false_positives(self):
        target = np.array([0, 1, 255, 255])
        # The final invalid prediction is never scored because truth is ignored.
        prediction = np.array([0, 1, 2, 999])
        counts = confusion_matrix(target, prediction, 3)
        result = metrics_from_confusion(counts, CLASSES, ignored_pixels=2)
        self.assertEqual(counts.tolist(), [[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        self.assertEqual(result["foreground_macro_iou"], 1.0)
        self.assertIsNone(result["per_class"][2]["iou"])
        self.assertEqual(result["scored_pixels"], 2)
        self.assertEqual(result["ignored_pixels"], 2)
        self.assertEqual(result["total_pixels"], 4)

    def test_all_ignored_metrics_have_no_defined_accuracy(self):
        counts = confusion_matrix(np.array([255, 255]), np.array([0, 1]), 3)
        result = metrics_from_confusion(counts, CLASSES, ignored_pixels=2)
        self.assertEqual(result["scored_pixels"], 0)
        self.assertIsNone(result["pixel_accuracy"])
        self.assertIsNone(result["foreground_macro_iou"])

    def test_background_accuracy_does_not_hide_missed_foreground(self):
        truth = np.array([0] * 99 + [1], dtype=np.int64)
        result = metrics_from_confusion(confusion_matrix(truth, np.zeros(100, dtype=np.int64), 3), CLASSES)
        self.assertEqual(result["pixel_accuracy"], 0.99)
        self.assertEqual(result["foreground_macro_iou"], 0.0)
        self.assertIsNone(result["per_class"][2]["iou"])
        self.assertEqual(result["foreground_classes_scored"], 1)

    def test_false_positive_absent_class_counts_as_zero(self):
        result = metrics_from_confusion(np.array([[2, 0, 1], [0, 2, 0], [0, 0, 0]]), CLASSES)
        self.assertEqual(result["per_class"][1]["iou"], 1.0)
        self.assertEqual(result["per_class"][2]["iou"], 0.0)
        self.assertEqual(result["foreground_macro_iou"], 0.5)

    def test_metrics_preserve_source_and_structure_mapping(self):
        counts = confusion_matrix(np.array([0, 1, 1, 2]), np.array([0, 1, 2, 2]), 3)
        result = metrics_from_confusion(counts, CLASSES)
        self.assertEqual(counts.tolist(), [[1, 0, 0], [0, 1, 1], [0, 0, 1]])
        self.assertEqual(result["per_class"][1]["sourceId"], 5)
        self.assertEqual(result["per_class"][1]["structureId"], "gallbladder")
        self.assertEqual(result["per_class"][1]["iou"], 0.5)
        self.assertAlmostEqual(result["per_class"][1]["dice"], 2 / 3)

    def test_invalid_source_ids_cannot_silently_be_used_as_training_indices(self):
        with self.assertRaises(ValueError):
            confusion_matrix(np.array([0, 5]), np.array([0, 1]), 3)
        with self.assertRaises(ValueError):
            confusion_matrix(np.array([0.0, 1.0]), np.array([0, 1]), 3)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"), "Training dependencies not installed")
class TrainingDataTests(unittest.TestCase):
    def test_balanced_weights_use_only_selected_resized_training_pixels(self):
        import torch
        from PIL import Image
        from holospex_ml.training import train

        class FixedLogits(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([0., 1., -1.]))

            def forward(self, pixels):
                logits = self.logits.view(1, 3, 1, 1).expand(pixels.shape[0], 3, *pixels.shape[-2:])
                return {"out": logits, "aux": logits}

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            samples = []
            masks = [
                np.array([[0, 0, 5, 255], [0, 0, 5, 4]], dtype=np.uint8),
                np.full((2, 4), 4, dtype=np.uint8),
                np.array([[0, 5, 4, 4], [0, 5, 4, 4]], dtype=np.uint8),
                np.full((2, 4), 5, dtype=np.uint8),
            ]
            for video, (split, mask) in enumerate(zip(["train", "train", "val", "test"], masks), start=1):
                image_path, mask_path = directory / f"{video}.jpg", directory / f"{video}.png"
                Image.fromarray(np.full((2, 4, 3), 127, dtype=np.uint8)).save(image_path)
                Image.fromarray(mask).save(mask_path)
                samples.append({"split": split, "videoId": video, "imagePath": str(image_path), "maskPath": str(mask_path)})
            manifest = {"dataset": "synthetic-weighting-test", "classes": CLASSES, "samples": samples, "ignoreIndex": 255, "ignoreSourceIds": [255]}
            model = FixedLogits()
            with patch("holospex_ml.training.build_model", return_value=model):
                report = train(manifest, directory / "run", epochs=1, batch_size=1, width=8, height=4,
                               device="cpu", pretrained=False, limit_train=1, class_weighting="balanced")
            details = report["config"]["class_weighting_details"]
            self.assertEqual(details["counts"], [16, 8, 4])
            self.assertEqual(details["ignored_pixels"], 4)
            self.assertEqual(details["sample_count"], 1)
            raw = 1 / np.sqrt(np.array([16, 8, 4]) / 28)
            expected_weights = np.clip(raw / raw[1:].mean(), 0.1, 5.0).astype(np.float32)
            np.testing.assert_array_equal(details["weights"], expected_weights)
            checkpoint = torch.load(report["checkpoint"], weights_only=True)
            self.assertEqual(checkpoint["class_weights"], details["weights"])
            self.assertEqual(checkpoint["class_weighting"], "balanced")

            # Both training heads use the weights; validation remains plain CE.
            train_target = torch.tensor([[0, 0, 1, 255], [0, 0, 1, 2]]).repeat_interleave(2, 0).repeat_interleave(2, 1).unsqueeze(0)
            initial_logits = torch.tensor([0., 1., -1.]).view(1, 3, 1, 1).expand(1, 3, 4, 8)
            expected_train = 1.4 * torch.nn.functional.cross_entropy(initial_logits, train_target, weight=torch.tensor(expected_weights), ignore_index=255)
            self.assertAlmostEqual(report["history"][0]["train_loss"], expected_train.item(), places=5)
            val_target = torch.tensor([[0, 1, 2, 2], [0, 1, 2, 2]]).repeat_interleave(2, 0).repeat_interleave(2, 1).unsqueeze(0)
            final_logits = model.logits.view(1, 3, 1, 1).expand(1, 3, 4, 8)
            expected_val = torch.nn.functional.cross_entropy(final_logits, val_target)
            weighted_val = torch.nn.functional.cross_entropy(final_logits, val_target, weight=torch.tensor(expected_weights))
            self.assertNotAlmostEqual(expected_val.item(), weighted_val.item(), places=3)
            self.assertAlmostEqual(report["history"][0]["validation"]["loss"], expected_val.item(), places=5)

    def test_balanced_weights_are_capped_and_missing_classes_fail(self):
        import torch
        from holospex_ml.training import _balanced_class_weights

        class TargetDataset:
            def __init__(self, counts):
                self.classes = [{"index": index, "structureId": f"class-{index}"} for index in range(len(counts))]
                self.target = torch.tensor(np.repeat(np.arange(len(counts)), counts))

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return torch.empty(0), self.target

        details = _balanced_class_weights(TargetDataset([10000] * 6 + [1]))
        self.assertEqual(details["weights"][-1], 5.0)
        self.assertAlmostEqual(min(details["weights"]), 0.1, places=6)
        with self.assertRaisesRegex(ValueError, "class-2.*Use more training samples"):
            _balanced_class_weights(TargetDataset([10, 2, 0]))

    def test_paired_resize_preserves_ignored_regions_without_background_relabeling(self):
        from PIL import Image
        from holospex_ml.training import SegmentationDataset

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            Image.fromarray(np.full((2, 3, 3), 127, dtype=np.uint8)).save(directory / "image.png")
            Image.fromarray(np.array([[0, 255, 5], [0, 255, 5]], dtype=np.uint8)).save(directory / "mask.png")
            sample = {"imagePath": str(directory / "image.png"), "maskPath": str(directory / "mask.png")}
            dataset = SegmentationDataset([sample], CLASSES, 6, 4, ignore_source_ids=[255])
            _, target = dataset[0]
            self.assertEqual(target[0].tolist(), [0, 0, 255, 255, 1, 1])
            self.assertEqual(int((target == 255).sum()), 8)
            with self.assertRaises(ValueError):
                SegmentationDataset([sample], CLASSES, 6, 4)[0]

    def test_evaluation_loss_and_counts_exclude_ignored_pixels(self):
        import torch
        from holospex_ml.training import _evaluate_model

        class FixedPrediction(torch.nn.Module):
            def forward(self, image):
                logits = torch.tensor([[[[2., -2.]], [[-2., 2.]], [[-2., -2.]]]])
                return {"out": logits}

        target = torch.tensor([[[0, 255]]])
        result = _evaluate_model(FixedPrediction(), [(torch.zeros(1, 3, 1, 2), target)], CLASSES, torch.device("cpu"))
        self.assertEqual(result["scored_pixels"], 1)
        self.assertEqual(result["ignored_pixels"], 1)
        self.assertEqual(result["per_class"][1]["predicted_pixels"], 0)
        self.assertTrue(np.isfinite(result["loss"]))
        with self.assertRaisesRegex(ValueError, "only ignored"):
            _evaluate_model(FixedPrediction(), [(torch.zeros(1, 3, 1, 2), torch.full((1, 1, 2), 255))], CLASSES, torch.device("cpu"))

    def test_evaluation_requires_same_explicit_ignore_policy(self):
        from holospex_ml.training import _check_ignore_policy

        manifest = {"classes": CLASSES, "ignoreIndex": 255, "ignoreSourceIds": [255]}
        self.assertEqual(_check_ignore_policy(manifest, {"ignore_index": 255, "ignore_source_ids": [255]}), [255])
        with self.assertRaisesRegex(ValueError, "policies differ"):
            _check_ignore_policy(manifest, {"ignore_index": 255, "ignore_source_ids": []})

    def test_singleton_batch_has_gradients_with_frozen_batchnorm(self):
        import torch
        from holospex_ml.model import build_model, freeze_batchnorm

        model = build_model(num_classes=3, pretrained=False).train()
        freeze_batchnorm(model)
        output = model(torch.rand(1, 3, 32, 64))
        target = torch.ones((1, 32, 64), dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(output["out"], target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(tuple(output["out"].shape), (1, 3, 32, 64))
        self.assertIsNotNone(model.classifier[-1].weight.grad)
        self.assertTrue(all(not module.training for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)))

    def test_paired_resize_preserves_discrete_training_labels(self):
        from PIL import Image
        from holospex_ml.training import SegmentationDataset

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            Image.fromarray(np.full((4, 6, 3), 127, dtype=np.uint8)).save(directory / "image.png")
            Image.fromarray(np.array([[0, 0, 5, 5, 4, 4]] * 4, dtype=np.uint8)).save(directory / "mask.png")
            dataset = SegmentationDataset([{"imagePath": str(directory / "image.png"), "maskPath": str(directory / "mask.png")}], CLASSES, 12, 8)
            pixels, target = dataset[0]
            self.assertEqual(tuple(pixels.shape), (3, 8, 12))
            self.assertEqual(tuple(target.shape), (8, 12))
            self.assertEqual(set(target.unique().tolist()), {0, 1, 2})
            self.assertEqual(target[0].tolist(), [0] * 4 + [1] * 4 + [2] * 4)

    def test_training_rejects_video_leakage_before_initializing_model(self):
        from holospex_ml.training import _validate_manifest

        with self.assertRaisesRegex(ValueError, "more than one split"):
            _validate_manifest({"classes": CLASSES, "samples": [{"split": "train", "videoId": "7"}, {"split": "val", "videoId": "7"}]})

    def test_relabeling_a_previous_training_case_as_test_is_rejected(self):
        from holospex_ml.training import evaluate

        # This new manifest is internally disjoint, but video7 already trained
        # the checkpoint. Merely checking the new manifest would miss leakage.
        manifest = {"classes": CLASSES, "samples": [
            {"split": "train", "videoId": 8}, {"split": "val", "videoId": 9},
            {"split": "test", "videoId": 7},
        ]}
        checkpoint = {"classes": CLASSES, "training": {"train_video_ids": ["7"], "val_video_ids": ["9"]}}
        with patch("holospex_ml.training.load_checkpoint", return_value=(None, checkpoint)):
            with self.assertRaisesRegex(ValueError, "overlaps checkpoint"):
                evaluate(Path("unused.pt"), manifest, split="test", device="cpu")

    def test_test_excludes_selection_cases_and_val_excludes_training_cases(self):
        from holospex_ml.training import _check_evaluation_split

        checkpoint = {"training": {"train_video_ids": ["7"], "val_video_ids": ["9"]}}
        for split, video in [("test", 7), ("test", 9), ("val", 7)]:
            with self.subTest(split=split, video=video):
                with self.assertRaisesRegex(ValueError, "overlaps checkpoint"):
                    _check_evaluation_split([{"videoId": video}], split, checkpoint)
        _check_evaluation_split([{"videoId": 9}], "val", checkpoint)
        _check_evaluation_split([{"videoId": 10}], "test", checkpoint)
        with self.assertRaisesRegex(ValueError, "cannot be verified"):
            _check_evaluation_split([{"videoId": 10}], "test", {"training": {}})

    def test_epoch_identity_and_exact_training_provenance_survive_safe_loading(self):
        import torch
        from PIL import Image
        from holospex_ml.training import _manifest_digest, train

        class TinySegmentation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Conv2d(3, len(CLASSES), 1)

            def forward(self, pixels):
                return {"out": self.head(pixels)}

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            samples = []
            for video, split in [(1, "train"), (2, "train"), (10, "val")]:
                image_path, mask_path = directory / f"{video}.jpg", directory / f"{video}.png"
                Image.fromarray(np.full((4, 6, 3), 127, dtype=np.uint8)).save(image_path)
                Image.fromarray(np.array([[0, 255, 5, 5, 4, 4]] * 4, dtype=np.uint8)).save(mask_path)
                samples.append({"split": split, "videoId": video, "imagePath": str(image_path), "maskPath": str(mask_path)})
            manifest = {"dataset": "synthetic-test-only", "classes": CLASSES, "samples": samples, "ignoreIndex": 255, "ignoreSourceIds": [255]}
            with patch("holospex_ml.training.build_model", side_effect=lambda *args, **kwargs: TinySegmentation()):
                with patch("holospex_ml.training._evaluate_model", side_effect=[{"foreground_macro_iou": 0.4}, {"foreground_macro_iou": 0.2}]):
                    report = train(manifest, directory / "run", epochs=2, batch_size=1, width=6, height=4,
                                   device="cpu", pretrained=False, limit_train=1)
            best = torch.load(report["checkpoint"], map_location="cpu", weights_only=True)
            last = torch.load(report["last_checkpoint"], map_location="cpu", weights_only=True)
            self.assertEqual(best["epochs_trained"], 1)
            self.assertEqual(last["epochs_trained"], 2)
            self.assertNotEqual(best["model_version"], last["model_version"])
            self.assertEqual(best["run_id"], last["run_id"])
            self.assertEqual(best["training"]["train_video_ids"], ["1"])
            self.assertEqual(best["training"]["val_video_ids"], ["10"])
            self.assertEqual(best["ignore_index"], 255)
            self.assertEqual(best["ignore_source_ids"], [255])
            self.assertEqual(best["training"]["ignore_source_ids"], [255])
            self.assertEqual(best["class_weighting"], "none")
            self.assertIsNone(best["class_weights"])
            self.assertIsNone(best["training"]["class_weighting_details"])
            self.assertEqual(report["history"][0]["train_ignored_pixels"], 4)
            self.assertEqual(best["training"]["manifest_sha256"], _manifest_digest(manifest))
            changed = {**manifest, "samples": list(reversed(samples))}
            self.assertNotEqual(_manifest_digest(manifest), _manifest_digest(changed))
            self.assertTrue(all(type(value) is str for value in best["training"]["software_versions"].values()))


if __name__ == "__main__":
    unittest.main()
