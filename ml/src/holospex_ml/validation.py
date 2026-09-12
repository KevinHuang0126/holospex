"""Validate exports using the same canonical JSON Schema as the web client.

Do not introduce a second hand-maintained Python schema. Changes to the shared
contract belong in contracts/schemas and must be coordinated with the web owner.
The geometric and temporal invariants below supplement JSON Schema, which
cannot compare a coordinate to another property of its containing frame.
"""

import json
import math
from pathlib import Path
from typing import Any

from jsonschema.validators import validator_for


class ContractError(ValueError):
    """The payload is unsuitable for downstream rendering."""


def default_schema_path() -> Path:
    """Resolve the monorepo contract; use --schema outside a source checkout."""
    return Path(__file__).resolve().parents[3] / "contracts/schemas/frame-result.schema.json"


def load_json(path: Path) -> Any:
    # Python's json parser otherwise accepts NaN/Infinity, which are not JSON
    # and can silently break bounds checks or browser rendering.
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _reject_constant(value: str) -> None:
    raise ContractError(f"Non-JSON numeric constant: {value}")


def _check_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{path}: numeric values must be finite")
    if isinstance(value, dict):
        for key, child in value.items():
            _check_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_finite(child, f"{path}[{index}]")


def validate_frame_result(data: Any, schema_path: Path | None = None) -> None:
    """Raise ContractError for invalid schema or cross-field semantics."""
    _check_finite(data)
    schema = load_json(schema_path or default_schema_path())
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    errors = list(validator_class(schema).iter_errors(data))
    if errors:
        messages = []
        for error in errors:
            location = "$" + "".join(
                f"[{part}]" if isinstance(part, int) else f".{part}"
                for part in error.absolute_path
            )
            messages.append(f"{location}: {error.message}")
        raise ContractError("\n".join(messages))

    instance_ids: set[str] = set()
    for index, structure in enumerate(data["structures"]):
        instance_id = structure["instanceId"]
        if instance_id in instance_ids:
            raise ContractError(f"$.structures[{index}].instanceId: duplicate {instance_id!r}")
        instance_ids.add(instance_id)
        for point_index, (x, y) in enumerate(structure["polygon"]):
            if not (0 <= x <= data["width"] and 0 <= y <= data["height"]):
                raise ContractError(
                    f"$.structures[{index}].polygon[{point_index}]: "
                    "coordinate is outside the original image bounds"
                )
    if data["source"] == "propagated_prediction":
        if data["propagatedFromTimestampMs"] >= data["timestampMs"]:
            raise ContractError("$.propagatedFromTimestampMs: must precede timestampMs")


def validate_export(data: Any, schema_path: Path | None = None) -> int:
    """Validate a single result or a JSON array of results; return frame count.

    A bundle is deliberately just an array for now. It may contain multiple
    media IDs or reviewed and predicted results for the same timestamp.
    """
    frames = data if isinstance(data, list) else [data]
    for index, frame in enumerate(frames):
        try:
            validate_frame_result(frame, schema_path)
        except ContractError as error:
            if isinstance(data, list):
                raise ContractError(f"Bundle frame {index}: {error}") from error
            raise
    return len(frames)
