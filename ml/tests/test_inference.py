import unittest
from pathlib import Path
import tempfile
from unittest import mock

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


@unittest.skipUnless(AVAILABLE, 'Install ml[train] for inference latency tests')
class InferenceExecutionTests(unittest.TestCase):
    def make_model(self):
        class TinySegmentationModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Conv2d(3, 2, 1)
                self.aux_classifier = torch.nn.Conv2d(3, 2, 1)
                self.load_events = []
                self.aux_calls = 0
                with torch.no_grad():
                    self.classifier.weight.zero_()
                    self.classifier.bias.zero_()
                    self.classifier.weight[1, 0, 0, 0] = 6
                    self.classifier.bias[1] = -3

            def load_state_dict(self, state, strict=True, **kwargs):
                self.load_events.append({'strict': strict, 'aux_present': self.aux_classifier is not None})
                return super().load_state_dict(state, strict=strict, **kwargs)

            def forward(self, tensor):
                result = {'out': self.classifier(tensor)}
                if self.aux_classifier is not None:
                    self.aux_calls += 1
                    result['aux'] = self.aux_classifier(tensor)
                return result

        return TinySegmentationModel()

    def checkpoint(self):
        return {
            'format_version': 1, 'architecture': 'deeplabv3_mobilenet_v3_large',
            'model_state': self.make_model().state_dict(),
            'input_size': {'width': 6, 'height': 4},
            'normalization': {'mean': [0, 0, 0], 'std': [1, 1, 1]},
            'classes': MaskExportTests.classes,
            'model_id': 'test-inference', 'model_version': 'fixture',
        }

    def frame_and_rgb(self):
        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        rgb[2:7, 2:10, 0] = 255
        frame = FrameInput(media_id='latency-fixture', frame_number=3, timestamp_ms=100,
                           width=12, height=8)
        return frame, rgb

    def assert_details_equal(self, expected, actual):
        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        self.assertEqual(expected[2], actual[2])

    def test_auxiliary_disabled_after_strict_checkpoint_load(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.pt'
            torch.save(self.checkpoint(), path)
            with mock.patch('holospex_ml.model.build_model', return_value=model):
                adapter = SegmentationAdapter(path, device='cpu')
        self.assertEqual(model.load_events, [{'strict': True, 'aux_present': True}])
        self.assertIsNone(adapter.model.aux_classifier)
        self.assertIn('aux_classifier.weight', adapter.checkpoint['model_state'])

    def test_missing_auxiliary_tensor_still_fails_strict_loading(self):
        model = self.make_model()
        checkpoint = self.checkpoint()
        del checkpoint['model_state']['aux_classifier.weight']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.pt'
            torch.save(checkpoint, path)
            with mock.patch('holospex_ml.model.build_model', return_value=model):
                with self.assertRaisesRegex(RuntimeError, 'aux_classifier.weight'):
                    SegmentationAdapter(path, device='cpu')
        self.assertEqual(model.load_events, [{'strict': True, 'aux_present': True}])
        self.assertIsNotNone(model.aux_classifier)

    def test_auxiliary_opt_out_preserves_logits_and_export_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.pt'
            torch.save(self.checkpoint(), path)
            with mock.patch('holospex_ml.model.build_model', side_effect=lambda **_: self.make_model()):
                baseline = SegmentationAdapter(path, device='cpu', min_area=1, run_auxiliary_head=True)
                optimized = SegmentationAdapter(path, device='cpu', min_area=1)
        frame, rgb = self.frame_and_rgb()
        captured_logits = []
        def capture_main_logits(module, inputs, output):
            captured_logits.append(output['out'].clone())
        baseline_hook = baseline.model.register_forward_hook(capture_main_logits)
        optimized_hook = optimized.model.register_forward_hook(capture_main_logits)
        try:
            expected = baseline.predict_rgb_details(frame, rgb)
            actual = optimized.predict_rgb_details(frame, rgb)
        finally:
            baseline_hook.remove()
            optimized_hook.remove()
        self.assert_details_equal(expected, actual)
        self.assertTrue(torch.equal(*captured_logits))
        self.assertEqual(baseline.model.aux_calls, 1)
        self.assertEqual(optimized.model.aux_calls, 0)
        self.assertTrue(actual[0]['structures'])

    def test_stage_timings_preserve_outputs_and_only_profiled_calls_synchronize(self):
        adapter = SegmentationAdapter.__new__(SegmentationAdapter)
        adapter.model, adapter.device = self.make_model().eval(), torch.device('cpu')
        adapter.threshold, adapter.min_area = 0.5, 1
        adapter.checkpoint = self.checkpoint()
        frame, rgb = self.frame_and_rgb()
        timings = {}
        with mock.patch('holospex_ml.inference._synchronize_device') as synchronize:
            expected = adapter.predict_rgb_details(frame, rgb)
            synchronize.assert_not_called()
            actual = adapter.predict_rgb_details(frame, rgb, timings=timings)
            self.assertEqual(synchronize.call_count, 5)
            self.assertTrue(all(call.args == (adapter.device,) for call in synchronize.call_args_list))
        self.assert_details_equal(expected, actual)
        self.assertEqual(set(timings), {
            'preprocess_ms', 'input_transfer_ms', 'forward_ms', 'output_resize_softmax_ms',
            'output_transfer_ms', 'geometry_ms', 'validation_ms',
        })
        self.assertTrue(all(np.isfinite(value) and value >= 0 for value in timings.values()))

    def test_synchronization_uses_requested_backend_and_cuda_index(self):
        from holospex_ml.inference import _synchronize_device
        with mock.patch('torch.cuda.synchronize') as cuda_sync, mock.patch('torch.mps.synchronize') as mps_sync:
            _synchronize_device(torch.device('cpu'))
            cuda_sync.assert_not_called()
            mps_sync.assert_not_called()
            cuda_device = torch.device('cuda:2')
            _synchronize_device(cuda_device)
            cuda_sync.assert_called_once_with(cuda_device)
            mps_sync.assert_not_called()
            _synchronize_device(torch.device('mps'))
            mps_sync.assert_called_once_with()
