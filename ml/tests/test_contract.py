from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema.exceptions import SchemaError

from holospex_ml import validation
from holospex_ml.adapters import FrameInput, UnconfiguredAdapter
from holospex_ml.cli import main
from holospex_ml.validation import ContractError, load_json, validate_export, validate_frame_result


def prediction() -> dict:
    """Synthetic test geometry, never a clinical annotation or model output."""
    result = UnconfiguredAdapter().predict(FrameInput("test-only", 0, 1000, 320, 240))
    result.update(
        status="ok",
        structures=[{
            "instanceId": "test-1",
            "structureId": "gallbladder",
            "polygon": [[0, 0], [320, 0], [0, 240]],
            "confidence": 0.5,
            "visibility": "visible",
        }],
    )
    result.pop("statusReason")
    return result


class ContractTests(unittest.TestCase):
    def test_unconfigured_adapter_is_explicitly_unsupported(self) -> None:
        result = UnconfiguredAdapter().predict(FrameInput("test-only", 0, 0, 320, 240))
        validate_frame_result(result)
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["structures"], [])
        self.assertIn("No inference model", result["statusReason"])

    def test_empty_successful_detection_is_distinct_from_failure(self) -> None:
        result = prediction()
        result["structures"] = []
        validate_frame_result(result)
        result["status"] = "error"
        with self.assertRaises(ContractError):
            validate_frame_result(result)

    def test_failed_result_cannot_carry_stale_overlays(self) -> None:
        result = prediction()
        result.update(status="error", statusReason="Inference failed")
        with self.assertRaises(ContractError):
            validate_frame_result(result)

    def test_prediction_requires_provenance_and_confidence(self) -> None:
        for absent in ("model", "confidence"):
            result = prediction()
            if absent == "model":
                result.pop("model")
            else:
                result["structures"][0].pop("confidence")
            with self.assertRaises(ContractError):
                validate_frame_result(result)

    def test_original_image_bounds_and_duplicate_instances(self) -> None:
        result = prediction()
        validate_frame_result(result)  # Points on width/height boundaries are legal.
        result["structures"][0]["polygon"][1][0] = 321
        with self.assertRaisesRegex(ContractError, "outside"):
            validate_frame_result(result)
        result = prediction()
        result["structures"].append(deepcopy(result["structures"][0]))
        with self.assertRaisesRegex(ContractError, "duplicate"):
            validate_frame_result(result)

    def test_propagation_must_refer_to_an_earlier_timestamp(self) -> None:
        result = prediction()
        result.update(source="propagated_prediction", propagatedFromTimestampMs=1000)
        with self.assertRaisesRegex(ContractError, "precede"):
            validate_frame_result(result)
        result["propagatedFromTimestampMs"] = 999
        validate_frame_result(result)

    def test_nonfinite_numbers_rejected_before_schema_validation(self) -> None:
        result = prediction()
        result["timestampMs"] = float("nan")
        with self.assertRaisesRegex(ContractError, "finite"):
            validate_frame_result(result)

    def test_bundle_reports_failed_frame(self) -> None:
        with self.assertRaisesRegex(ContractError, "Bundle frame 1"):
            validate_export([prediction(), {}])

    def test_cached_and_uncached_validation_have_identical_semantics(self) -> None:
        cases = [prediction()]
        for change in (
            lambda frame: frame.pop("model"),
            lambda frame: frame.update(timestampMs=float("nan")),
            lambda frame: frame["structures"][0]["polygon"][1].__setitem__(0, 321),
            lambda frame: frame["structures"].append(deepcopy(frame["structures"][0])),
            lambda frame: frame.update(source="propagated_prediction", propagatedFromTimestampMs=1000),
            lambda frame: frame.update(status="error", statusReason="Inference failed"),
        ):
            frame = prediction()
            change(frame)
            cases.append(frame)
        for index, frame in enumerate(cases):
            outcomes = []
            for use_cache in (False, True):
                with self.subTest(case=index, use_cache=use_cache):
                    try:
                        validate_frame_result(frame, use_cache=use_cache)
                        outcomes.append(None)
                    except ContractError as error:
                        outcomes.append(str(error))
            self.assertEqual(outcomes[0], outcomes[1])
            if index:
                self.assertIsNotNone(outcomes[0])
            else:
                self.assertIsNone(outcomes[0])
        for use_cache in (False, True):
            with self.assertRaisesRegex(ContractError, "Bundle frame 1"):
                validate_export([prediction(), {}], use_cache=use_cache)

    def test_validator_reuses_schema_and_can_bypass_cache_for_benchmarks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            schema_path = Path(directory) / "schema.json"
            schema_path.write_bytes(validation.default_schema_path().read_bytes())
            with patch.object(validation, "load_json", wraps=load_json) as load:
                validate_export([prediction(), prediction()], schema_path)
                self.assertEqual(load.call_count, 1)
                validate_export([prediction(), prediction()], schema_path, use_cache=False)
                self.assertEqual(load.call_count, 3)

    def test_schema_cache_tracks_explicit_paths_edits_and_atomic_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            schema_path = Path(directory) / "schema.json"
            other_path = Path(directory) / "other.json"
            schema = load_json(validation.default_schema_path())
            schema["properties"]["mediaId"]["const"] = "test-only"
            accepted = json.dumps(schema)
            schema["properties"]["mediaId"]["const"] = "different"
            rejected = json.dumps(schema)
            self.assertEqual(len(accepted), len(rejected))
            schema_path.write_text(accepted)
            other_path.write_text(rejected)
            validate_frame_result(prediction(), schema_path)
            with self.assertRaises(ContractError):
                validate_frame_result(prediction(), other_path)

            original_stat = schema_path.stat()
            schema_path.write_text(rejected)
            os.utime(schema_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            with self.assertRaises(ContractError):
                validate_frame_result(prediction(), schema_path)

            replacement = Path(directory) / "replacement.json"
            replacement.write_text(accepted)
            os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            replacement.replace(schema_path)
            validate_frame_result(prediction(), schema_path)

            schema_path.unlink()
            with self.assertRaises(FileNotFoundError):
                validate_frame_result(prediction(), schema_path)

    def test_cache_does_not_hide_an_invalid_schema_edit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            schema_path = Path(directory) / "schema.json"
            schema_path.write_bytes(validation.default_schema_path().read_bytes())
            validate_frame_result(prediction(), schema_path)
            schema_path.write_text('{"type": "invalid-type"}')
            for use_cache in (True, False):
                with self.assertRaises(SchemaError):
                    validate_frame_result(prediction(), schema_path, use_cache=use_cache)

    def test_cli_export_is_valid_and_will_not_replace_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frame.json"
            arguments = ["export-unconfigured", str(output), "--media-id", "test-only", "--width", "320", "--height", "240"]
            self.assertEqual(main(arguments), 0)
            validate_frame_result(load_json(output))
            original = output.read_bytes()
            self.assertEqual(main(arguments), 1)
            self.assertEqual(output.read_bytes(), original)

    def test_cli_invalid_input_creates_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid.json"
            result = main(["export-unconfigured", str(output), "--media-id", "test-only", "--width", "0", "--height", "240"])
            self.assertEqual(result, 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
