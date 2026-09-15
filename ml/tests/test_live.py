"""Runner boundary tests with explicit fake predictors, never model evidence."""

from base64 import b64encode
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from http.client import HTTPConnection
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from holospex_ml import current_model
from holospex_ml.adapters import FrameInput
from holospex_ml.live import (
    MAX_IMAGE_PIXELS, MAX_REQUEST_BYTES, create_server, decode_request, load_predictor,
    main, resolve_configuration,
)

MODEL = {"id": "test-predictor-only", "version": "unit-test"}
FRAME = {"mediaId": "test-camera", "frameNumber": 17, "timestampMs": 321.5,
         "width": 4, "height": 3}


def encoded_image(image_format="JPEG", orientation=None):
    output = BytesIO()
    with Image.new("RGB", (4, 3), (130, 60, 30)) as image:
        options = {}
        if orientation is not None:
            exif = Image.Exif()
            exif[274] = orientation
            options["exif"] = exif
        image.save(output, format=image_format, **options)
    return b64encode(output.getvalue()).decode("ascii")


def request_body(**updates):
    value = {"frame": deepcopy(FRAME), "imageBase64": encoded_image()}
    value.update(updates)
    return json.dumps(value).encode()


def fake_prediction(frame, image, minimum_confidence=None):
    return {"schemaVersion": "1.0.0", "mediaId": frame.media_id,
            "frameNumber": frame.frame_number, "timestampMs": frame.timestamp_ms,
            "width": frame.width, "height": frame.height,
            "coordinateSpace": "original_pixels", "source": "ml_prediction",
            "status": "ok", "model": dict(MODEL), "structures": []}


@contextmanager
def running_server(predictor=fake_prediction, token=None):
    server = create_server("127.0.0.1", 0, predictor, MODEL, 0.61, token)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def request(port, method="POST", body=None, headers=None, path="/identify"):
    connection = HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request(method, path, body, headers or {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read()), dict(response.getheaders())
    finally:
        connection.close()


class LiveIdentificationTests(unittest.TestCase):
    def test_ready_and_prediction_preserve_original_capture_identity_without_files(self):
        captured = []

        def predict(frame, image, minimum_confidence):
            captured.append((frame, image.mode, image.size, image.getpixel((0, 0))))
            result = fake_prediction(frame, image)
            result["structures"] = [{"instanceId": "test-only", "structureId": "gallbladder",
                                     "polygon": [[0, 0], [4, 0], [0, 3]],
                                     "confidence": 0.7, "visibility": "partial"}]
            return result

        with running_server(predict) as port:
            status, ready, headers = request(port, "GET")
            self.assertEqual(status, 200)
            self.assertEqual(ready, {"status": "ready", "model": MODEL,
                                     "minimumConfidence": 0.61, "supportsMinimumConfidence": True,
                                     "dataset": "Endoscapes-Seg50"})
            self.assertEqual(headers["Cache-Control"], "no-store")
            status, result, _ = request(port, body=request_body())
            self.assertEqual(status, 200)
            self.assertEqual({key: result[key] for key in FRAME}, FRAME)
            self.assertEqual(result["structures"][0]["polygon"], [[0, 0], [4, 0], [0, 3]])
            self.assertEqual(captured[0][0].image_path, None)
            self.assertEqual(captured[0][1:3], ("RGB", (4, 3)))

    def test_rejects_malformed_frames_and_jpegs_before_predictor(self):
        calls = []
        invalid = [b"{}", b'{"frame": NaN}', request_body(imageBase64="!!!!"),
                   request_body(imageBase64="data:image/jpeg;base64," + encoded_image()),
                   request_body(imageBase64=encoded_image("PNG")),
                   request_body(imageBase64=encoded_image(orientation=6)),
                   request_body(imageBase64=encoded_image()[:-4]),
                   b'{"frame":{},"frame":{},"imageBase64":""}']
        for key, value in [("width", 5), ("width", True), ("frameNumber", -1),
                           ("frameNumber", 2**53), ("timestampMs", float("inf")),
                           ("timestampMs", True), ("mediaId", ""), ("mediaId", "x\n")]:
            frame = dict(FRAME)
            frame[key] = value
            invalid.append(request_body(frame=frame))
        frame = dict(FRAME, width=4096, height=MAX_IMAGE_PIXELS // 4096 + 1)
        invalid.append(request_body(frame=frame))
        with running_server(lambda frame, image, minimum_confidence: calls.append(frame)) as port:
            for body in invalid:
                with self.subTest(body=body[:70]):
                    status, result, _ = request(port, body=body)
                    self.assertEqual(status, 400)
                    self.assertEqual(result["status"], "error")
        self.assertEqual(calls, [])

    def test_body_and_content_type_limits_apply_before_reading_or_inference(self):
        with running_server() as port:
            status, _, _ = request(port, body=b"", headers={"Content-Type": "application/json",
                                     "Content-Length": str(MAX_REQUEST_BYTES + 1)})
            self.assertEqual(status, 413)
            status, _, _ = request(port, body=b"{}", headers={"Content-Type": "text/plain"})
            self.assertEqual(status, 415)
            status, _, _ = request(port, body=b"{}", headers={"Content-Type": "application/json",
                                     "Transfer-Encoding": "chunked"})
            self.assertEqual(status, 400)
        with self.assertRaises(ValueError):
            decode_request(b" " * (MAX_REQUEST_BYTES + 1))

        # Pillow may reject a malicious declared JPEG size before our size check.
        with patch.object(Image, "open", side_effect=Image.DecompressionBombError("test limit")):
            with running_server() as port:
                self.assertEqual(request(port, body=request_body())[0], 400)

    def test_request_cutoff_accepts_boundaries_and_does_not_change_server_default(self):
        cutoffs = []

        def predict(frame, image, minimum_confidence):
            cutoffs.append(minimum_confidence)
            return fake_prediction(frame, image)

        with running_server(predict) as port:
            for values in ({}, {"minimumConfidence": 0}, {"minimumConfidence": 1},
                           {"minimumConfidence": 0.83}, {}):
                with self.subTest(values=values):
                    self.assertEqual(request(port, body=request_body(**values))[0], 200)
            self.assertEqual(request(port, "GET")[1]["minimumConfidence"], 0.61)
        self.assertEqual(cutoffs, [0.61, 0, 1, 0.83, 0.61])

    def test_invalid_request_cutoff_is_rejected_before_inference(self):
        predict = Mock(side_effect=fake_prediction)
        with running_server(predict) as port:
            for value in (None, True, False, "0.5", "", [], {}, -0.01, 1.01,
                          float("nan"), float("inf"), -float("inf")):
                with self.subTest(value=value):
                    status, result, _ = request(port, body=request_body(minimumConfidence=value))
                    self.assertEqual(status, 400)
                    self.assertEqual(result["status"], "error")
            self.assertEqual(request(port, body=request_body())[0], 200)
        self.assertEqual(predict.call_count, 1, "invalid values do not run or change the model")

    def test_busy_inference_drops_new_requests_but_health_remains_available(self):
        entered, release = threading.Event(), threading.Event()
        calls, responses = [], []

        def blocked_predictor(frame, image, minimum_confidence):
            calls.append(frame)
            entered.set()
            self.assertTrue(release.wait(3))
            return fake_prediction(frame, image)

        with running_server(blocked_predictor) as port:
            first = threading.Thread(target=lambda: responses.append(request(port, body=request_body())))
            first.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(request(port, body=request_body())[0], 429)
                self.assertEqual(request(port, "GET")[0], 200)
                self.assertEqual(len(calls), 1)
            finally:
                release.set()
                first.join(3)
            self.assertEqual(responses[0][0], 200)
            self.assertEqual(request(port, body=request_body())[0], 200)
            self.assertEqual(len(calls), 2)

    def test_failed_or_mismatched_predictions_are_withheld_and_lock_recovers(self):
        for change in ({"frameNumber": 18}, {"width": 5}, {"source": "reviewed_annotation"},
                       {"model": {"id": "other", "version": "1"}},
                       {"structures": [{"invalid": "geometry"}]}):
            calls = []

            def predict(frame, image, minimum_confidence):
                result = fake_prediction(frame, image)
                if not calls:
                    result.update(change)
                calls.append(frame)
                return result

            with self.subTest(change=change), running_server(predict) as port:
                status, error, _ = request(port, body=request_body())
                self.assertEqual(status, 500)
                self.assertEqual(error, {"status": "error", "message": "Identification unavailable."})
                self.assertEqual(request(port, body=request_body())[0], 200)

        def fails(frame, image, minimum_confidence):
            raise RuntimeError("sensitive checkpoint path or frame must not escape")

        with running_server(fails) as port:
            self.assertNotIn("sensitive", json.dumps(request(port, body=request_body())[1]))

    def test_optional_token_protects_health_and_identification_and_nonloopback_requires_it(self):
        with self.assertRaisesRegex(ValueError, "Non-loopback"):
            create_server("0.0.0.0", 0, fake_prediction, MODEL, 0.61)
        for threshold in (None, True, -0.1, float("nan"), 1.1):
            with self.assertRaises(ValueError):
                create_server("127.0.0.1", 0, fake_prediction, MODEL, threshold)
        with running_server(token="test-token-only") as port:
            self.assertEqual(request(port, "GET")[0], 401)
            self.assertEqual(request(port, body=request_body())[0], 401)
            headers = {"Authorization": "Bearer test-token-only", "Content-Type": "application/json"}
            self.assertEqual(request(port, "GET", headers=headers)[0], 200)
            self.assertEqual(request(port, body=request_body(), headers=headers)[0], 200)
            self.assertEqual(request(port, "GET", headers=headers, path="/identify/other")[0], 404)
            self.assertEqual(request(port, "OPTIONS")[0], 405)

    def test_unavailable_predictions_remain_empty_and_have_canonical_status(self):
        def predict(frame, image, minimum_confidence):
            result = fake_prediction(frame, image)
            result.update(status="unsupported", statusReason="Test-only unsupported input")
            return result

        with running_server(predict) as port:
            status, result, _ = request(port, body=request_body())
            self.assertEqual(status, 200)
            self.assertEqual(result["status"], "unsupported")
            self.assertEqual(result["structures"], [])

    def test_checkpoint_adapter_is_loaded_once_and_reuses_in_memory_original_frames(self):
        classes = ("background", "gallbladder", "cystic_duct", "cystic_artery", "cystic_plate",
                   "hepatocystic_triangle_dissection", "tool")
        instances, received = [], []

        class FakeAdapter:
            def __init__(self, checkpoint, device, threshold):
                instances.append((checkpoint, device, threshold))
                self.checkpoint = {"model_id": MODEL["id"], "model_version": MODEL["version"],
                                   "dataset": {"dataset": "endoscapes-seg50"},
                                   "classes": [{"structureId": value} for value in classes]}

            def predict_rgb_details(self, frame, rgb, *, threshold=None):
                received.append((frame, rgb, threshold))
                return fake_prediction(frame, rgb), None, None

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "test-only.pt"
            checkpoint.write_bytes(b"fake adapter loading test, not weights")
            with patch.dict("sys.modules", {
                "numpy": SimpleNamespace(asarray=lambda image: image),
                "holospex_ml.dataset": SimpleNamespace(STRUCTURE_ORDER=classes),
                "holospex_ml.inference": SimpleNamespace(SegmentationAdapter=FakeAdapter),
            }):
                predict, model = load_predictor(checkpoint, "cpu", 0.61)
                frame, image, cutoff = decode_request(request_body())
                with image:
                    predict(frame, image)
                    predict(frame, image, 0.82)
                    predict(frame, image)
                self.assertEqual(model, MODEL)
                self.assertEqual(instances, [(checkpoint.resolve(), "cpu", 0.61)])
                self.assertEqual(len(received), 3)
                self.assertEqual([item[2] for item in received], [None, 0.82, None])
                self.assertIsNone(cutoff)
                self.assertEqual(received[0][0], FrameInput("test-camera", 17, 321.5, 4, 3))
            self.assertEqual(list(Path(directory).iterdir()), [checkpoint])


class CurrentModelTests(unittest.TestCase):
    @contextmanager
    def fixture_selection(self):
        """Use tiny explicit fake bytes so tests never require or imitate weights."""
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "current" / "best.pt"
            checkpoint.parent.mkdir()
            content = b"test-only checkpoint fingerprint, not trained weights"
            checkpoint.write_bytes(content)
            with patch.multiple(current_model, CHECKPOINT_PATH=checkpoint,
                                CHECKPOINT_BYTES=len(content),
                                CHECKPOINT_SHA256=hashlib.sha256(content).hexdigest()):
                yield checkpoint, content

    def test_default_selection_and_threshold_are_independent_of_working_directory(self):
        with self.fixture_selection() as (checkpoint, _):
            self.assertEqual(resolve_configuration(None, None), (checkpoint.resolve(), 0.5))
            self.assertEqual(resolve_configuration(checkpoint, None), (checkpoint.resolve(), 0.5))
            self.assertEqual(resolve_configuration(checkpoint, 0.65), (checkpoint.resolve(), 0.65))
            custom = checkpoint.parent / "other.pt"
            self.assertEqual(resolve_configuration(custom, 0.61), (custom.resolve(), 0.61))
            for threshold in (None, -0.1, True, float("nan"), 1.1):
                with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                    resolve_configuration(custom, threshold)
        self.assertTrue(current_model.CHECKPOINT_PATH.is_absolute())

    def test_altered_or_incomplete_current_weights_are_rejected_before_adapter_load(self):
        with self.fixture_selection() as (checkpoint, content):
            current_model.verify_checkpoint(checkpoint)
            with patch.dict("sys.modules", {"holospex_ml.inference": None}):
                for invalid in (content[:-1], b"x" + content[1:]):
                    checkpoint.write_bytes(invalid)
                    with self.subTest(content=invalid), self.assertRaisesRegex(ValueError, "Checkpoint"):
                        load_predictor(device="cpu")
            checkpoint.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                current_model.verify_checkpoint(checkpoint)

    def test_promoted_metadata_and_preprocessing_bind_identity_to_in_memory_results(self):
        classes = ("background", "gallbladder", "cystic_duct", "cystic_artery", "cystic_plate",
                   "hepatocystic_triangle_dissection", "tool")
        metadata = {
            "architecture": current_model.ARCHITECTURE,
            "model_id": current_model.MODEL_ID, "model_version": current_model.MODEL_VERSION,
            "input_size": deepcopy(current_model.INPUT_SIZE),
            "normalization": deepcopy(current_model.NORMALIZATION),
            "dataset": {"dataset": "endoscapes-seg50+reviewed-partial+reviewed-partial"},
            "classes": [{"structureId": value} for value in classes],
        }
        instances = []
        mutate_on_load = []

        class FakeAdapter:
            def __init__(self, checkpoint, device, threshold):
                self.checkpoint = deepcopy(metadata)
                instances.append((checkpoint, device, threshold))
                if mutate_on_load:
                    checkpoint.write_bytes(b"x" + checkpoint.read_bytes()[1:])

            def predict_rgb_details(self, frame, image, *, threshold=None):
                result = fake_prediction(frame, image)
                result["model"] = {"id": self.checkpoint["model_id"],
                                   "version": self.checkpoint["model_version"]}
                return result, None, None

        with self.fixture_selection() as (checkpoint, _), patch.dict("sys.modules", {
            "numpy": SimpleNamespace(asarray=lambda image: image),
            "holospex_ml.dataset": SimpleNamespace(STRUCTURE_ORDER=classes),
            "holospex_ml.inference": SimpleNamespace(SegmentationAdapter=FakeAdapter),
        }):
            predict, model = load_predictor(device="cpu")
            with Image.new("RGB", (4, 3)) as image:
                frame = FrameInput("test-camera", 17, 321.5, 4, 3)
                first, second = predict(frame, image), predict(frame, image)
            self.assertEqual(instances, [(checkpoint.resolve(), "cpu", 0.5)])
            self.assertEqual(model, {"id": current_model.MODEL_ID, "version": current_model.MODEL_VERSION})
            self.assertEqual(first["model"], model)
            self.assertEqual(first, second)
            self.assertEqual({key: first[key] for key in FRAME}, FRAME)

            # Preparation appends one exact suffix per reviewed batch; these
            # remain the same seven anatomy channels, not a different task.
            for dataset_name in ("endoscapes-seg50", "endoscapes-seg50+reviewed-partial",
                                 "endoscapes-seg50+reviewed-partial+reviewed-partial"):
                metadata["dataset"] = {"dataset": dataset_name}
                with self.subTest(dataset=dataset_name):
                    self.assertEqual(load_predictor(device="cpu")[1], model)

            # Even an adapter returning plausible output cannot advertise the
            # promoted selection if its identity or preprocessing differs.
            for field, value in (("model_version", "old-model"),
                                 ("model_id", "holospex-deeplabv3-mobilenetv3"),
                                 ("architecture", "deeplabv3_mobilenet_v3_large"),
                                 ("input_size", {"width": 896, "height": 512}),
                                 ("normalization", {"mean": [0, 0, 0], "std": [1, 1, 1]}),
                                 ("dataset", {"dataset": "other"}),
                                 ("dataset", {"dataset": "endoscapes-seg50+unreviewed"}),
                                 ("dataset", {"dataset": "endoscapes-seg50+reviewed-partial+other"}),
                                 ("classes", [{"structureId": value} for value in reversed(classes)])):
                previous = metadata[field]
                metadata[field] = value
                try:
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        load_predictor(device="cpu")
                finally:
                    metadata[field] = previous
            mutate_on_load.append(True)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_predictor(device="cpu")

    def test_default_startup_advertises_loaded_identity_and_custom_threshold_is_required(self):
        server = Mock()
        server.serve_forever.side_effect = KeyboardInterrupt
        output, error = StringIO(), StringIO()
        with self.fixture_selection() as (checkpoint, _), \
                patch("holospex_ml.live.load_predictor", return_value=(fake_prediction, MODEL)) as load, \
                patch("holospex_ml.live.create_server", return_value=server) as create, \
                patch("holospex_ml.live.os.environ", {}), \
                patch("sys.stdout", output), patch("sys.stderr", error):
            self.assertEqual(main(["--device", "cpu"]), 0)
            load.assert_called_once_with(checkpoint.resolve(), "cpu", 0.5)
            create.assert_called_once_with("127.0.0.1", 8765, fake_prediction, MODEL, 0.5, None)
            server.server_close.assert_called_once()
            self.assertIn(MODEL["version"], output.getvalue())
            self.assertEqual(main(["--checkpoint", str(checkpoint.parent / "custom.pt")]), 1)
            self.assertIn("threshold", error.getvalue())
            self.assertEqual(load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
