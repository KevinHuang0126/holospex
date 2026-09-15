"""Exercise the latency protocol with tiny synthetic inputs, never real weights."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    from PIL import Image
    import torch
    from holospex_ml import benchmark, inference
    from holospex_ml.adapters import UnconfiguredAdapter
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


CLASSES = [{"index": 0, "structureId": "background"},
           {"index": 1, "structureId": "gallbladder"}]


def tiny_adapter_class(*, mismatch=False):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.zeros(()))
            self.aux_classifier = torch.nn.Identity()

        def forward(self, rgb):
            zero = rgb[:, :1] * 0 + self.offset
            background = zero + (2 if mismatch and self.aux_classifier is None else 0)
            return {"out": torch.cat((background, zero + 1), dim=1)}

    class TinyAdapter:
        instances = []

        def __init__(self, checkpoint_path, *, device, run_auxiliary_head):
            self.model = TinyModel()
            self.device = torch.device("cpu")
            self.threshold, self.min_area = 0.5, 1
            self.checkpoint = {"model_id": "synthetic-protocol-test", "model_version": "0",
                               "input_size": {"width": 4, "height": 3}, "classes": CLASSES}
            self.calls = []
            self.instances.append(self)

        def predict_rgb_details(self, frame, rgb, *, timings=None):
            self.calls.append((self.model.aux_classifier is not None,
                               inference.validate_frame_result.keywords["use_cache"]))
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float()
            with torch.inference_mode():
                logits = self.model(tensor)["out"]
            labels = logits.argmax(1)[0].numpy().astype(np.uint8)
            result = UnconfiguredAdapter().predict(frame)
            inference.validate_frame_result(result)
            if timings is not None:
                timings.update(model_ms=0.01, validation_ms=0.02)
            return result, labels, {"small": 0, "holes": 0}

    return TinyAdapter


@unittest.skipUnless(AVAILABLE, "Install ml[train] for latency protocol tests")
class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        original_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, original_threads)

    def sample(self, video, frame, *, split="val"):
        path = self.root / f"{split}-{video}-{frame}.png"
        Image.fromarray(np.full((3, 4, 3), frame, dtype=np.uint8)).save(path)
        return {"videoId": video, "frameNumber": frame, "timestampMs": frame * 100,
                "split": split, "imagePath": str(path)}

    def inputs(self):
        samples = [self.sample(2, 1), self.sample(1, 2)]
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps({"samples": samples, "classes": CLASSES}))
        checkpoint_path = self.root / "fake-checkpoint.pt"
        checkpoint_path.write_bytes(b"Protocol fixture only; no trained weights.")
        return checkpoint_path, manifest_path

    def test_sample_selection_is_deterministic_and_validation_only(self):
        samples = [self.sample(2, 1), self.sample(1, 2), self.sample(1, 1)]
        for split in ("train", "test"):
            samples.append({"videoId": 0, "frameNumber": 0, "timestampMs": 0,
                            "split": split, "imagePath": str(self.root / "must-not-open.png")})
        loaded, records = benchmark._load_samples({"samples": samples})
        self.assertEqual([(row["video_id"], row["frame_number"]) for row in records],
                         [("1", 1), ("1", 2), ("2", 1)])
        reversed_loaded, _ = benchmark._load_samples({"samples": list(reversed(samples))}, limit=2)
        self.assertEqual([frame for frame, _ in reversed_loaded], [frame for frame, _ in loaded[:2]])
        for (frame, rgb), record in zip(loaded, records):
            self.assertEqual(rgb.shape, (3, 4, 3))
            self.assertEqual((frame.width, frame.height), (4, 3))
            self.assertEqual(record["image_sha256"], hashlib.sha256(frame.image_path.read_bytes()).hexdigest())

    def test_sample_selection_rejects_duplicates_empty_validation_and_bad_limit(self):
        sample = self.sample(1, 1)
        duplicate = dict(sample, videoId="1")
        with self.assertRaisesRegex(ValueError, "Duplicate validation frame"):
            benchmark._load_samples({"samples": [sample, duplicate]})
        with self.assertRaisesRegex(ValueError, "no validation samples"):
            benchmark._load_samples({"samples": [dict(sample, split="test")]})
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            benchmark._load_samples({"samples": [sample]}, limit=0)

    def test_latency_summary_uses_linear_percentile_and_rejects_invalid_samples(self):
        summary = benchmark._summary([0, 10, 20, 30])
        self.assertEqual({key: value for key, value in summary.items() if key != "p95_ms"},
                         {"count": 4, "median_ms": 15.0, "min_ms": 0.0, "max_ms": 30.0})
        self.assertAlmostEqual(summary["p95_ms"], 28.5)
        self.assertEqual(benchmark._summary([0])["p95_ms"], 0)
        for values in ([], [-1], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                benchmark._summary(values)

    def test_variant_restores_auxiliary_head_and_validator_on_exception(self):
        adapter = tiny_adapter_class()(None, device="cpu", run_auxiliary_head=True)
        original_auxiliary = adapter.model.aux_classifier
        alternate_auxiliary = torch.nn.Identity()
        original_validator = inference.validate_frame_result
        for name, config in benchmark.VARIANTS.items():
            with self.subTest(variant=name):
                with self.assertRaisesRegex(RuntimeError, "deliberate fixture failure"):
                    with benchmark._variant(adapter, alternate_auxiliary, name):
                        self.assertIs(adapter.model.aux_classifier,
                                      alternate_auxiliary if config["auxiliary_head"] else None)
                        self.assertEqual(inference.validate_frame_result.keywords["use_cache"],
                                         config["cached_validation"])
                        raise RuntimeError("deliberate fixture failure")
                self.assertIs(adapter.model.aux_classifier, original_auxiliary)
                self.assertIs(inference.validate_frame_result, original_validator)

    def test_four_way_benchmark_persists_raw_counts_parity_and_refuses_overwrite(self):
        checkpoint, manifest = self.inputs()
        output = self.root / "completed-run"
        adapter_class = tiny_adapter_class()
        with patch.object(inference, "SegmentationAdapter", adapter_class), \
                patch.object(benchmark.platform, "system", return_value="SyntheticTest"):
            report = benchmark.benchmark_latency(checkpoint, manifest, output, device="cpu",
                                                  repeats=2, warmup=1, profile_frames=1, cpu_threads=1)
        saved = json.loads((output / "benchmark.json").read_text())
        self.assertEqual(saved, report)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(set(report["variants"]), {"baseline", "cached_validation", "no_auxiliary_head", "combined"})
        self.assertEqual(len(report["calls"]), 16)
        self.assertEqual(Counter(row["variant"] for row in report["calls"]),
                         {name: 4 for name in benchmark.VARIANTS})
        self.assertEqual(Counter((row["repeat"], row["sample_index"]) for row in report["calls"]),
                         {(repeat, sample): 4 for repeat in range(2) for sample in range(2)})
        self.assertEqual(len(report["profiles"]), 4)
        self.assertEqual(len(report["logit_parity"]), 2)
        self.assertTrue(report["parity"]["timed_outputs_equal"])
        self.assertTrue(report["parity"]["main_logits_and_outputs_equal"])
        for row in report["calls"]:
            self.assertGreaterEqual(row["latency_ms"], 0)
            self.assertEqual(row["labels_shape"], [3, 4])
            self.assertEqual(len(row["labels_sha256"]), 64)
        for row in report["logit_parity"]:
            self.assertEqual(row["baseline"]["logits_shape"], [1, 2, 3, 4])
            self.assertEqual(len(row["baseline"]["logits_sha256"]), 64)
        self.assertEqual(report["summary"]["baseline"]["count"], 4)
        self.assertEqual([row["combined"]["count"] for row in report["repeat_summary"]], [2, 2])
        self.assertEqual(set(adapter_class.instances[0].calls), {(True, False), (True, True),
                                                                (False, False), (False, True)})
        self.assertTrue((output / "RESULTS.md").is_file())
        original = (output / "benchmark.json").read_bytes()
        with patch.object(inference, "SegmentationAdapter") as constructor:
            with self.assertRaises(FileExistsError):
                benchmark.benchmark_latency(checkpoint, manifest, output, device="cpu")
            constructor.assert_not_called()
        self.assertEqual((output / "benchmark.json").read_bytes(), original)

    def test_output_mismatch_persists_failed_report_and_never_writes_success_summary(self):
        checkpoint, manifest = self.inputs()
        output = self.root / "mismatch-run"
        with patch.object(inference, "SegmentationAdapter", tiny_adapter_class(mismatch=True)), \
                patch.object(benchmark.platform, "system", return_value="SyntheticTest"):
            with self.assertRaisesRegex(RuntimeError, "Output parity failed"):
                benchmark.benchmark_latency(checkpoint, manifest, output, device="cpu",
                                              repeats=1, warmup=1, profile_frames=1, cpu_threads=1)
        report = json.loads((output / "benchmark.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["type"], "RuntimeError")
        self.assertFalse(report["parity"]["timed_outputs_equal"])
        self.assertFalse(report["parity"]["main_logits_and_outputs_equal"])
        self.assertTrue(report["parity"]["mismatches"])
        self.assertEqual(len(report["calls"]), 8)
        self.assertFalse((output / "RESULTS.md").exists())

    def test_denied_optional_hardware_metadata_does_not_abort_benchmark(self):
        checkpoint, manifest = self.inputs()
        with patch.object(inference, "SegmentationAdapter", tiny_adapter_class()), \
                patch.object(benchmark.platform, "system", return_value="Darwin"), \
                patch.object(benchmark.subprocess, "run", side_effect=PermissionError("denied")):
            report = benchmark.benchmark_latency(checkpoint, manifest, self.root / "denied-metadata",
                                                  device="cpu", repeats=1, warmup=1,
                                                  profile_frames=1, cpu_threads=1)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["hardware"])
        self.assertTrue(all(value is None for value in report["hardware"].values()))


if __name__ == "__main__":
    unittest.main()
