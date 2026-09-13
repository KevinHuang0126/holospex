import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(package) is not None for package in ("numpy", "PIL", "torch", "torchvision"))
if AVAILABLE:
    import numpy as np
CLASSES = [{"index": index, "sourceId": source, "structureId": structure} for index, (source, structure) in enumerate([
    (0, "background"), (5, "gallbladder"), (4, "cystic_duct"),
    (3, "cystic_artery"), (1, "cystic_plate"),
    (2, "hepatocystic_triangle_dissection"), (6, "tool"),
])]


@unittest.skipUnless(AVAILABLE, "Install ml[train] for original-grid evaluation tests")
class OriginalEvaluationTests(unittest.TestCase):
    def test_logits_resize_before_argmax_changes_boundary(self):
        import torch
        from holospex_ml.evaluation import _logits_to_original_labels

        logits = torch.tensor([[[[10., 0.]], [[0., 1.]]]])
        prediction = _logits_to_original_labels(logits, (1, 4), 2)
        np.testing.assert_array_equal(prediction, [[0, 0, 0, 1]])
        nearest_labels = torch.nn.functional.interpolate(logits.argmax(1, keepdim=True).float(), size=(1, 4), mode="nearest")
        self.assertNotEqual(prediction.tolist(), nearest_labels[0, 0].tolist())

    def test_case_equal_aggregates_within_video_before_averaging(self):
        from holospex_ml.evaluation import _case_equal, _report_confusion

        # Ten perfect frames in A and one missed frame in B: equal-case score
        # must be 0.5, even though global pixels yield 10/11.
        a, b = np.zeros((7, 7), dtype=np.int64), np.zeros((7, 7), dtype=np.int64)
        a[2, 2], b[2, 0] = 1000, 100
        equal = _case_equal([_report_confusion(a, CLASSES), _report_confusion(b, CLASSES)], CLASSES)
        self.assertEqual(equal["per_class"][2]["iou"], 0.5)
        self.assertEqual(equal["small_anatomy_macro_iou"], 0.5)
        self.assertEqual(equal["per_class"][2]["present_truth_case_count"], 2)
        self.assertAlmostEqual(_report_confusion(a + b, CLASSES)["per_class"][2]["iou"], 10 / 11)

    def test_absent_truth_false_positives_count_as_zero_but_absent_both_is_undefined(self):
        from holospex_ml.evaluation import _case_equal, _report_confusion

        correct, fp, absent = [np.zeros((7, 7), dtype=np.int64) for _ in range(3)]
        correct[2, 2], fp[0, 2], absent[0, 0] = 10, 5, 10
        equal = _case_equal([_report_confusion(c, CLASSES) for c in (correct, fp, absent)], CLASSES)
        duct = equal["per_class"][2]
        self.assertEqual(duct["iou"], 0.5)
        self.assertEqual(duct["dice"], 0.5)
        self.assertEqual(duct["present_truth_case_count"], 1)
        self.assertEqual(duct["false_positive_only_case_count"], 1)
        self.assertEqual(duct["absent_both_case_count"], 1)
        self.assertEqual(duct["cases_scored"], 2)
        self.assertIsNone(equal["per_class"][3]["iou"])

    def fixture(self, directory):
        import torch
        from PIL import Image

        class FixedLogits(torch.nn.Module):
            def forward(self, image):
                self.seen_input = image.detach().clone()
                result = torch.full((1, 7, 1, 2), -10., device=image.device)
                result[0, 0, 0] = torch.tensor([10., 0.], device=image.device)
                result[0, 2, 0] = torch.tensor([0., 1.], device=image.device)
                return {"out": result}

        image, mask = directory / "frame.png", directory / "mask.png"
        Image.fromarray(np.full((1, 4, 3), 128, dtype=np.uint8)).save(image)
        Image.fromarray(np.array([[0, 255, 0, 4]], dtype=np.uint8)).save(mask)
        sample = {"split": "val", "videoId": 2, "frameNumber": 5, "timestampMs": 200,
                  "imagePath": str(image), "maskPath": str(mask)}
        manifest = {"classes": CLASSES, "samples": [sample], "ignoreIndex": 255, "ignoreSourceIds": [255]}
        checkpoint = {
            "classes": CLASSES, "input_size": {"width": 2, "height": 1},
            "normalization": {"mean": [0.1, 0.2, 0.3], "std": [0.5, 0.5, 0.5]},
            "model_id": "test-model", "model_version": "synthetic-epoch-1",
            "ignore_index": 255, "ignore_source_ids": [255],
            "training": {"train_video_ids": ["1"], "val_video_ids": ["2"],
                         "limited_run": False, "manifest_sha256": "training-manifest-digest"},
        }
        path = directory / "fixture.pt"
        path.write_bytes(b"mock-checkpoint-identity")
        return FixedLogits(), checkpoint, manifest, path

    def test_end_to_end_original_mask_ignore_mapping_normalization_and_identity(self):
        from holospex_ml.evaluation import evaluate_original
        from holospex_ml.training import _manifest_digest

        with tempfile.TemporaryDirectory() as temporary:
            model, checkpoint, manifest, path = self.fixture(Path(temporary))
            with patch("holospex_ml.evaluation.load_checkpoint", return_value=(model, checkpoint)):
                report = evaluate_original(path, manifest, device="cpu")
        self.assertEqual(report["split"], "val")
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["scored_pixels"], 3)
        self.assertEqual(report["ignored_pixels"], 1)
        self.assertEqual(report["per_class"][2]["support_pixels"], 1)
        self.assertEqual(report["per_class"][2]["iou"], 1)
        self.assertEqual(report["small_anatomy_macro_iou"], 1)
        self.assertEqual(report["per_class"][2]["precision"], 1)
        self.assertEqual(report["per_class"][2]["recall"], 1)
        self.assertEqual(report["per_class"][2]["present_truth_case_count"], 1)
        self.assertEqual(report["checkpoint_sha256"], hashlib.sha256(b"mock-checkpoint-identity").hexdigest())
        self.assertEqual(report["evaluation_manifest_sha256"], _manifest_digest(manifest))
        self.assertEqual(report["training_manifest_sha256"], "training-manifest-digest")
        self.assertEqual(report["metric_resolution"]["sizes"], [{"width": 4, "height": 1}])
        self.assertEqual(report["evaluation_video_ids"], ["2"])
        self.assertEqual(tuple(model.seen_input.shape), (1, 3, 1, 2))
        np.testing.assert_allclose(model.seen_input[0, :, 0, 0].numpy(), (128 / 255 - np.array([0.1, 0.2, 0.3])) / .5, atol=1e-6)
        json.dumps(report, allow_nan=False)

    def test_evaluator_pools_multiple_frames_per_video_for_case_equal_scores(self):
        from holospex_ml.evaluation import evaluate_original

        with tempfile.TemporaryDirectory() as temporary:
            model, checkpoint, manifest, path = self.fixture(Path(temporary))
            first = manifest["samples"][0]
            manifest["samples"] = [
                {**first, "frameNumber": 0},
                {**first, "frameNumber": 1},
                {**first, "videoId": 3, "frameNumber": 0},
            ]
            original_forward = model.forward
            call_count = 0

            def predict(image):
                nonlocal call_count
                call_count += 1
                output = original_forward(image)
                if call_count == 3:
                    output["out"][:, 0] = 10  # Miss the duct in the second case.
                return output

            with patch("holospex_ml.evaluation.load_checkpoint", return_value=(model, checkpoint)), patch.object(model, "forward", side_effect=predict):
                report = evaluate_original(path, manifest, device="cpu")
        self.assertEqual([video["sample_count"] for video in report["per_video"]], [2, 1])
        self.assertEqual(report["case_count"], 2)
        self.assertAlmostEqual(report["per_class"][2]["iou"], 2 / 3)
        self.assertEqual(report["case_equal"]["per_class"][2]["iou"], .5)
        self.assertEqual(report["case_equal"]["small_anatomy_macro_iou"], .5)

    def test_heldout_provenance_and_ignore_policy_are_enforced_before_reading_frames(self):
        from holospex_ml.evaluation import evaluate_original

        with tempfile.TemporaryDirectory() as temporary:
            model, checkpoint, manifest, path = self.fixture(Path(temporary))
            variants = []
            overlap = copy.deepcopy(manifest)
            overlap["samples"][0]["videoId"] = 1
            variants.append((overlap, checkpoint, "val", "overlaps"))
            selection_overlap = copy.deepcopy(manifest)
            selection_overlap["samples"][0]["split"] = "test"
            variants.append((selection_overlap, checkpoint, "test", "overlaps"))
            missing = copy.deepcopy(checkpoint)
            del missing["training"]["train_video_ids"]
            variants.append((manifest, missing, "val", "lacks train_video_ids"))
            wrong_ignore = copy.deepcopy(checkpoint)
            wrong_ignore["ignore_source_ids"] = []
            variants.append((manifest, wrong_ignore, "val", "policies differ"))
            wrong_classes = copy.deepcopy(checkpoint)
            wrong_classes["classes"][2]["sourceId"] = 99
            variants.append((manifest, wrong_classes, "val", "class maps differ"))
            for data, metadata, split, message in variants:
                with self.subTest(message=message, split=split):
                    with patch("holospex_ml.evaluation.load_checkpoint", return_value=(model, metadata)), patch("holospex_ml.evaluation.read_image") as reader:
                        with self.assertRaisesRegex(ValueError, message):
                            evaluate_original(path, data, split=split, device="cpu")
                        reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
