import unittest
from dataclasses import replace
from pathlib import Path
import tempfile

try:
    import numpy as np
    import torch
    from PIL import Image
    from holospex_ml.adapters import FrameInput
    from holospex_ml.inference import SegmentationAdapter, _is_simple_polygon, masks_to_structures
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Install ml[train] for segmentation export tests')
class MaskExportTests(unittest.TestCase):
    classes = [dict(index=0, structureId='background'), dict(index=1, structureId='gallbladder')]

    def probabilities(self):
        probs = np.zeros((2, 30, 40), dtype=np.float32)
        probs[0] = 0.95
        probs[1] = 0.05
        probs[:, 5:25, 5:30] = np.array([0.1, 0.9])[:, None, None]
        return probs

    def test_original_frame_polygon_and_mean_score(self):
        structures, labels, withheld = masks_to_structures(self.probabilities(), self.classes, min_area=4)
        self.assertEqual(labels.shape, (30, 40))
        self.assertEqual(len(structures), 1)
        self.assertAlmostEqual(structures[0]['confidence'], 0.9, places=5)
        self.assertEqual(withheld['holes'], 0)
        self.assertEqual(structures[0]['structureId'], 'gallbladder')

    def test_low_confidence_withheld_without_inventing_unsupported_state(self):
        structures, _, _ = masks_to_structures(self.probabilities(), self.classes, threshold=0.99)
        self.assertEqual(structures, [])

    def test_holes_not_filled_by_simple_polygon_export(self):
        probabilities = self.probabilities()
        probabilities[:, 10:15, 10:15] = np.array([0.95, 0.05])[:, None, None]
        structures, _, withheld = masks_to_structures(probabilities, self.classes, min_area=4)
        self.assertEqual(structures, [])
        self.assertEqual(withheld['holes'], 1)

    def test_diagonal_contact_is_withheld_but_raw_mask_is_preserved(self):
        probabilities = np.zeros((2, 30, 40), dtype=np.float32)
        probabilities[0] = 1
        for rows, columns in ((slice(2, 12), slice(2, 12)), (slice(12, 22), slice(12, 22))):
            probabilities[0, rows, columns] = 0
            probabilities[1, rows, columns] = 1
        structures, raw, withheld = masks_to_structures(probabilities, self.classes, min_area=4)
        self.assertEqual(structures, [])
        self.assertEqual(withheld['nonSimple'], 1)
        self.assertEqual(withheld['holes'], 0)
        self.assertEqual(np.count_nonzero(raw == 1), 200)

    def test_separate_simple_components_still_export(self):
        probabilities = np.zeros((2, 30, 40), dtype=np.float32)
        probabilities[0] = 1
        for rows, columns in ((slice(2, 12), slice(2, 12)), (slice(13, 23), slice(13, 23))):
            probabilities[0, rows, columns] = 0
            probabilities[1, rows, columns] = 1
        structures, _, withheld = masks_to_structures(probabilities, self.classes, min_area=4)
        self.assertEqual(len(structures), 2)
        self.assertEqual(withheld['nonSimple'], 0)

    def test_crossing_and_touching_boundaries_are_not_simple(self):
        invalid = [
            [(0, 0), (10, 10), (0, 10), (10, 0)],  # crossing, no repeated vertices
            [(0, 0), (10, 0), (10, 10), (5, 0), (0, 10)],  # nonadjacent edge touch
            [(0, 0), (10, 0), (5, 0), (5, 10), (0, 10)],  # adjacent backtrack
        ]
        for polygon in invalid:
            with self.subTest(polygon=polygon):
                self.assertFalse(_is_simple_polygon(polygon))
        self.assertTrue(_is_simple_polygon([(0, 0), (10, 0), (10, 10), (5, 5), (0, 10)]))

    def test_unknown_probability_shape_rejected(self):
        with self.assertRaises(ValueError):
            masks_to_structures(np.zeros((20, 20)), self.classes)

    def test_decoded_rgb_and_image_file_follow_identical_preprocessing(self):
        class RedChannelModel:
            def __call__(self, tensor):
                return {'out': torch.cat((torch.zeros_like(tensor[:, :1]), tensor[:, :1] * 10), dim=1)}

        adapter = SegmentationAdapter.__new__(SegmentationAdapter)
        adapter.model, adapter.device = RedChannelModel(), torch.device('cpu')
        adapter.threshold, adapter.min_area = 0.5, 4
        adapter.checkpoint = {
            'input_size': {'width': 6, 'height': 4},
            'normalization': {'mean': [0, 0, 0], 'std': [1, 1, 1]},
            'classes': self.classes, 'model_id': 'test-red-channel', 'model_version': 'fixture',
        }
        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'red.png'
            Image.fromarray(rgb).save(path)
            frame = FrameInput(media_id='fixture', frame_number=0, timestamp_ms=0, width=12, height=8, image_path=path)
            from_file = adapter.predict_details(frame)
            from_rgb = adapter.predict_rgb_details(frame, rgb)
        self.assertEqual(from_file[0], from_rgb[0])
        np.testing.assert_array_equal(from_file[1], from_rgb[1])
        self.assertTrue((from_rgb[1] == 1).all())
        self.assertEqual(from_rgb[1].shape, (8, 12))

    def test_per_call_cutoff_changes_export_without_mutating_default_or_raw_predictions(self):
        class ConstantModel:
            calls = 0

            def __call__(self, tensor):
                self.calls += 1
                # Explicit deterministic test output: foreground probability 0.8.
                logits = torch.zeros_like(tensor[:, :1])
                return {'out': torch.cat((logits, logits + np.log(4)), dim=1)}

        adapter = SegmentationAdapter.__new__(SegmentationAdapter)
        adapter.model, adapter.device = ConstantModel(), torch.device('cpu')
        adapter.threshold, adapter.min_area = 0.5, 4
        adapter.checkpoint = {
            'input_size': {'width': 6, 'height': 4},
            'normalization': {'mean': [0, 0, 0], 'std': [1, 1, 1]},
            'classes': self.classes, 'model_id': 'constant-test-output', 'model_version': 'fixture',
        }
        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        frame = FrameInput(media_id='fixture', frame_number=0, timestamp_ms=0, width=12, height=8)
        baseline = adapter.predict_rgb_details(frame, rgb)
        self.assertEqual(len(baseline[0]['structures']), 1)
        for threshold, expected_count in ((0.9, 0), (0, 1), (1, 0), (None, 1)):
            with self.subTest(threshold=threshold):
                result, raw, _ = adapter.predict_rgb_details(frame, rgb, threshold=threshold)
                self.assertEqual(len(result['structures']), expected_count)
                np.testing.assert_array_equal(raw, baseline[1])
                self.assertEqual(adapter.threshold, 0.5)
        with tempfile.TemporaryDirectory() as directory:
            file_frame = replace(frame, image_path=Path(directory) / 'fixture.png')
            Image.fromarray(rgb).save(file_frame.image_path)
            self.assertEqual(adapter.predict_details(file_frame)[0], baseline[0])

        calls = adapter.model.calls
        for threshold in (True, False, "0.5", [], -0.1, 1.1, float('nan'), float('inf')):
            with self.subTest(invalid=threshold), self.assertRaises(ValueError):
                adapter.predict_rgb_details(frame, rgb, threshold=threshold)
        self.assertEqual(adapter.model.calls, calls, 'invalid cutoffs fail before model inference')
