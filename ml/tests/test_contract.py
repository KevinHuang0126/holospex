from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

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
