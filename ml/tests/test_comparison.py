"""Synthetic comparisons verify selection and whole-frame scoring, not anatomy."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from holospex_ml.comparison import FOCUS_CLASSES, render_small_anatomy_comparison
from holospex_ml.dataset import STRUCTURE_ORDER


class SmallAnatomyComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.classes = [{"index": index, "sourceId": index, "structureId": name}
                        for index, name in enumerate(STRUCTURE_ORDER)]
        self.samples, self.truths, self.calls = [], {}, []
        for number, side in enumerate((1, 2, 3, 4), start=1):
            image_path, mask_path = self.root / f"{number}.png", self.root / f"{number}.mask.png"
            image = np.zeros((240, 320, 3), dtype=np.uint8)
            image[:, :, 0] = np.arange(320, dtype=np.uint16)[None, :] % 256
            truth = np.zeros((240, 320), dtype=np.uint8)
            truth[:10] = 255
            for position, focus in enumerate(FOCUS_CLASSES):
                index = STRUCTURE_ORDER.index(focus)
                truth[100:100 + side, 110 + position * 8:110 + position * 8 + side] = index
            Image.fromarray(image).save(image_path)
            Image.fromarray(truth).save(mask_path)
            self.truths[str(image_path)] = truth
            self.samples.append({"split": "val", "videoId": number, "frameNumber": number * 10,
                                 "timestampMs": number * 400, "imagePath": str(image_path), "maskPath": str(mask_path)})
        # This unreadable training path proves selection does not inspect train.
        self.samples.append({"split": "train", "videoId": 99, "frameNumber": 1,
                             "timestampMs": 40, "imagePath": "unreadable", "maskPath": "unreadable"})
        self.manifest = {"classes": self.classes, "samples": self.samples,
                         "ignoreIndex": 255, "ignoreSourceIds": [255]}
        self.before, self.after = self.root / "before.pt", self.root / "after.pt"
        self.before.write_bytes(b"synthetic before checkpoint")
        self.after.write_bytes(b"synthetic after checkpoint")
        test = self

        class FakeAdapter:
            def __init__(self, path, device):
                self.after = Path(path) == test.after
                self.checkpoint = {"classes": test.classes, "model_id": "synthetic",
                    "model_version": "after" if self.after else "before", "epochs_trained": 1,
                    "input_size": {"width": 160 if self.after else 80, "height": 120 if self.after else 60},
                    "ignore_index": 255, "ignore_source_ids": [255], "training": {"train_video_ids": [99]}}

            def predict_rgb_details(self, frame, image):
                test.calls.append((self.after, frame, image.shape))
                truth = test.truths[str(frame.image_path)]
                labels = np.zeros_like(truth)
                if self.after:
                    labels = np.where(truth == 255, 6, truth).astype(np.uint8)
                    # Errors outside the display crop must still lower IoU.
                    for offset, focus in enumerate(FOCUS_CLASSES):
                        labels[-1, offset] = STRUCTURE_ORDER.index(focus)
                return {}, labels, {}

        self.adapter = FakeAdapter

    def test_median_selection_whole_frame_inference_and_uncropped_scores(self):
        output = self.root / "comparison.png"
        with patch("holospex_ml.comparison.SegmentationAdapter", self.adapter):
            result = render_small_anatomy_comparison(self.before, self.after, self.manifest, output, "cpu")
        self.assertEqual(result, output.resolve())
        audit = json.loads(output.with_suffix(".json").read_text())
        self.assertEqual(len(audit["rows"]), 4)
        # Four classes choose the same median-area frame; infer it once/model.
        self.assertEqual(len(self.calls), 2)
        for _, frame, shape in self.calls:
            self.assertEqual(shape, (240, 320, 3))
            self.assertEqual((frame.width, frame.height, frame.frame_number), (320, 240, 30))
        for row in audit["rows"]:
            self.assertEqual(row["annotationAreaPixels"], 9)
            self.assertEqual(row["selectionCandidateCount"], 4)
            self.assertEqual(row["selectedSortedRankZeroBased"], 2)
            self.assertEqual(row["videoId"], 3)
            self.assertLess(row["cropBox"]["bottomExclusive"], 239)
            index = STRUCTURE_ORDER.index(row["focusClass"])
            before, after = (row["wholeFrameOriginalGridMetrics"][key] for key in ("before", "after"))
            self.assertEqual(before["per_class"][index]["iou"], 0.0)
            self.assertAlmostEqual(after["per_class"][index]["iou"], 0.9)
            self.assertEqual(after["ignored_pixels"], 3200)
            self.assertEqual(row["imageSha256"], hashlib.sha256(Path(row["imagePath"]).read_bytes()).hexdigest())
        self.assertEqual(audit["checkpoints"]["before"]["modelVersion"], "before")
        self.assertEqual(audit["checkpoints"]["after"]["sha256"], hashlib.sha256(self.after.read_bytes()).hexdigest())
        self.assertFalse(audit["display"]["confidenceThresholdApplied"])
        with Image.open(output) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.width, 1376)

    def test_existing_output_is_not_replaced_or_model_loaded(self):
        output = self.root / "comparison.png"
        output.with_suffix(".json").write_text("keep me")
        with patch("holospex_ml.comparison.SegmentationAdapter") as adapter:
            with self.assertRaises(FileExistsError):
                render_small_anatomy_comparison(self.before, self.after, self.manifest, output)
        adapter.assert_not_called()
        self.assertEqual(output.with_suffix(".json").read_text(), "keep me")

    def test_training_video_overlap_is_rejected(self):
        self.samples[0]["videoId"] = 99
        with patch("holospex_ml.comparison.SegmentationAdapter", self.adapter):
            with self.assertRaisesRegex(ValueError, "overlap"):
                render_small_anatomy_comparison(self.before, self.after, self.manifest, self.root / "comparison.png")
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
