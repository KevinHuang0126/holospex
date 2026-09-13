from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import av
    import numpy as np
    from holospex_ml.video import export_video
    from holospex_ml.validation import validate_export
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class FakeFrame:
    def __init__(self, pts, *, width=12, height=8, time_base=Fraction(1, 1000)):
        self.pts, self.time_base, self.width, self.height = pts, time_base, width, height

    def to_ndarray(self, format):
        assert format == "rgb24"
        return np.full((self.height, self.width, 3), 80, dtype=np.uint8)


class FakeContainer:
    def __init__(self, frames, decode_error=None):
        self.frames, self.decode_error = frames, decode_error
        self.streams = SimpleNamespace(video=[SimpleNamespace(index=0, time_base=Fraction(1, 1000))], audio=[])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def decode(self, stream):
        yield from self.frames
        if self.decode_error:
            raise self.decode_error


class MockAdapter:
    """Test fixture only; the test makes no surgical-prediction claim."""
    threshold, min_area = 0.5, 4
    checkpoint = {
        "model_id": "test-video-adapter", "model_version": "fixture",
        "classes": [{"index": 0, "structureId": "background"}, {"index": 1, "structureId": "gallbladder"}],
    }

    def __init__(self, *, fail_at=None, stale=False):
        self.frames, self.fail_at, self.stale = [], fail_at, stale

    def predict_rgb_details(self, frame, rgb):
        if frame.frame_number == self.fail_at:
            raise RuntimeError("test inference failure")
        self.frames.append(frame)
        assert rgb.shape == (frame.height, frame.width, 3)
        return ({
            "schemaVersion": "1.0.0", "mediaId": frame.media_id,
            "frameNumber": 0 if self.stale else frame.frame_number, "timestampMs": frame.timestamp_ms,
            "width": frame.width, "height": frame.height, "coordinateSpace": "original_pixels",
            "source": "ml_prediction", "status": "ok",
            "model": {"id": self.checkpoint["model_id"], "version": self.checkpoint["model_version"]},
            "structures": [],
        }, np.zeros(rgb.shape[:2], dtype=np.uint8), {"nonSimple": 0})


@unittest.skipUnless(AVAILABLE, "Install ml[train] and PyAV for video pipeline tests")
class VideoExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.input = self.root / "fixture.mp4"
        self.input.write_bytes(b"mock-video-fixture")
        self.output = self.root / "predictions.json"
        self.addCleanup(self.directory.cleanup)

    def run_export(self, frames, adapter=None, **kwargs):
        with patch("av.open", return_value=FakeContainer(frames)):
            return export_video(adapter or MockAdapter(), self.input, self.output, "test-clip", **kwargs)

    def test_variable_pts_preserves_zero_start_media_clock(self):
        adapter = MockAdapter()
        report = self.run_export([FakeFrame(0), FakeFrame(40), FakeFrame(120)], adapter)
        results = json.loads(self.output.read_text())
        self.assertEqual(validate_export(results), 3)
        self.assertEqual([result["timestampMs"] for result in results], [0, 40, 120])
        self.assertEqual([result["frameNumber"] for result in results], [0, 1, 2])
        self.assertEqual(report["clipPTSBase"], {"pts": 0, "timeBase": {"numerator": 1, "denominator": 1000}})
        self.assertEqual(report["inputSha256"], hashlib.sha256(self.input.read_bytes()).hexdigest())
        self.assertEqual((report["decodedFrameCount"], report["processedFrameCount"]), (3, 3))
        self.assertFalse(report["partialLimitReached"])
        self.assertTrue(all(frame.image_path is None for frame in adapter.frames))
        self.assertEqual(len(list(self.output.with_suffix(".masks").glob("*.png"))), 3)

    def test_partial_limit_counts_decoded_lookahead_without_inference(self):
        report = self.run_export([FakeFrame(0), FakeFrame(50), FakeFrame(130)], max_frames=2)
        self.assertEqual((report["decodedFrameCount"], report["processedFrameCount"]), (3, 2))
        self.assertTrue(report["partialLimitReached"])
        self.assertEqual(len(json.loads(self.output.read_text())), 2)

    def test_exact_end_at_limit_is_not_marked_partial(self):
        report = self.run_export([FakeFrame(0), FakeFrame(50)], max_frames=2)
        self.assertFalse(report["partialLimitReached"])

    def test_missing_or_nonmonotonic_pts_and_variable_dimensions_fail(self):
        bad_sequences = [
            [FakeFrame(None)], [FakeFrame(0), FakeFrame(0)],
            [FakeFrame(0), FakeFrame(-10)],
            [FakeFrame(0), FakeFrame(50, width=13)],
        ]
        for number, frames in enumerate(bad_sequences):
            with self.subTest(number=number):
                output = self.root / f"invalid-{number}.json"
                with patch("av.open", return_value=FakeContainer(frames)):
                    with self.assertRaises(ValueError):
                        export_video(MockAdapter(), self.input, output, "test-clip")
                self.assertFalse(output.exists())
                self.assertEqual(json.loads(output.with_suffix(".info.json").read_text())["status"], "failed")

    def test_nonzero_initial_pts_is_rejected_and_preserved_in_failure_audit(self):
        with self.assertRaisesRegex(ValueError, "setpts=PTS-STARTPTS"):
            self.run_export([FakeFrame(1000)])
        report = json.loads(self.output.with_suffix(".info.json").read_text())
        self.assertFalse(self.output.exists())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["clipPTSBase"]["pts"], 1000)
        self.assertEqual((report["decodedFrameCount"], report["processedFrameCount"]), (1, 0))

    def test_audio_track_is_rejected_with_preparation_instruction(self):
        container = FakeContainer([FakeFrame(0)])
        container.streams.audio = [SimpleNamespace(index=1)]
        with patch("av.open", return_value=container):
            with self.assertRaisesRegex(ValueError, "ffmpeg -an"):
                export_video(MockAdapter(), self.input, self.output, "test-clip")
        self.assertFalse(self.output.exists())
        report = json.loads(self.output.with_suffix(".info.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["audioStreamCount"], 1)
        self.assertEqual(report["decodedFrameCount"], 0)

    def test_inference_failure_does_not_publish_a_complete_array(self):
        with self.assertRaisesRegex(RuntimeError, "test inference failure"):
            self.run_export([FakeFrame(0), FakeFrame(50)], MockAdapter(fail_at=1))
        report = json.loads(self.output.with_suffix(".info.json").read_text())
        self.assertFalse(self.output.exists())
        self.assertEqual(report["status"], "failed")
        self.assertEqual((report["decodedFrameCount"], report["processedFrameCount"]), (2, 1))
        self.assertEqual(len(list(self.output.with_suffix(".masks").glob("*.png"))), 1)

    def test_decode_failure_is_reported_without_partial_array(self):
        with patch("av.open", return_value=FakeContainer([FakeFrame(0)], RuntimeError("decode failed"))):
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                export_video(MockAdapter(), self.input, self.output, "test-clip")
        self.assertFalse(self.output.exists())
        self.assertEqual(json.loads(self.output.with_suffix(".info.json").read_text())["processedFrameCount"], 1)

    def test_stale_adapter_result_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "different frame"):
            self.run_export([FakeFrame(0), FakeFrame(50)], MockAdapter(stale=True))
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_replaced(self):
        self.output.write_text("keep existing result")
        with self.assertRaises(FileExistsError):
            self.run_export([FakeFrame(0)])
        self.assertEqual(self.output.read_text(), "keep existing result")

    def test_actual_pyav_decoding_uses_presentation_times(self):
        video_path = self.root / "generated.mkv"
        with av.open(str(video_path), mode="w") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 12, 8, "bgr0"
            stream.time_base = Fraction(1, 1000)
            stream.codec_context.time_base = Fraction(1, 1000)
            for pts in (0, 40, 120):
                frame = av.VideoFrame.from_ndarray(np.zeros((8, 12, 3), dtype=np.uint8), format="rgb24")
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        report = export_video(MockAdapter(), video_path, self.output, "generated-test-clip")
        self.assertEqual([frame["timestampMs"] for frame in report["frames"]], [0, 40, 120])
        self.assertEqual(report["originalSize"], {"width": 12, "height": 8})


if __name__ == "__main__":
    unittest.main()
