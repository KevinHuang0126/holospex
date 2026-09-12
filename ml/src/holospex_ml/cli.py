"""Small offline CLI. Real inference is intentionally not implemented yet."""

import argparse
import json
from pathlib import Path
import sys

from .adapters import FrameInput, UnconfiguredAdapter
from .validation import ContractError, load_json, validate_export, validate_frame_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate one frame result or a JSON array of results")
    validate.add_argument("input", type=Path)
    validate.add_argument("--schema", type=Path, help="Override the canonical frame-result schema path")

    export = commands.add_parser("export-unconfigured", help="Export an unsupported result; does not run a model")
    export.add_argument("output", type=Path)
    export.add_argument("--media-id", required=True)
    export.add_argument("--width", required=True, type=int)
    export.add_argument("--height", required=True, type=int)
    export.add_argument("--frame-number", type=int, default=0)
    export.add_argument("--timestamp-ms", type=float, default=0)
    export.add_argument("--schema", type=Path, help="Override the canonical frame-result schema path")

    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            count = validate_export(load_json(args.input), args.schema)
            print(f"Valid: {count} frame result(s)")
        else:
            frame = FrameInput(
                media_id=args.media_id,
                frame_number=args.frame_number,
                timestamp_ms=args.timestamp_ms,
                width=args.width,
                height=args.height,
            )
            result = UnconfiguredAdapter().predict(frame)
            validate_frame_result(result, args.schema)
            # Validate before touching the filesystem. Exclusive creation avoids
            # accidentally replacing a real inference export with this stub.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(result, output, indent=2, allow_nan=False)
                output.write("\n")
            print(f"Wrote unsupported result (no predictions): {args.output}")
    except (ContractError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
