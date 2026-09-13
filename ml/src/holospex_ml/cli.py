"""Download, inspect, train, evaluate, and export the offline anatomy pipeline."""

import argparse
import json
from pathlib import Path
import sys
import importlib.metadata
import platform

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

    commands.add_parser("doctor", help="Show the current Python/ML environment and available devices")
    download = commands.add_parser("download-endoscapes", help="Fetch only Seg50 images/masks from the official public archive")
    download.add_argument("--output-dir", type=Path, default=Path("ml/data/endoscapes"))
    download.add_argument("--metadata-only", action="store_true")
    prepare = commands.add_parser("prepare-endoscapes", help="Validate dataset mapping/splits and write a training manifest")
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--fps", type=float, required=True, help="Explicit filename frame-number timebase; verify against your actual release/video")
    prepare.add_argument("--ignore-source-id", type=int, action="append", default=[], help="Explicit source pixel ID to exclude from supervision/metrics; official uncertain pixels use 255")
    prepare.add_argument("--exclude-frame", action="append", default=[], help="Explicit VIDEO_FRAME stem to quarantine after a documented data audit")
    preview = commands.add_parser("preview", help="Render images with dataset annotations for inspection")
    preview.add_argument("--manifest", type=Path, required=True)
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--split", choices=["train", "val", "test"], default="train")
    preview.add_argument("--limit", type=int, default=6)
    train = commands.add_parser("train", help="Fine-tune the anatomy baseline on train; select checkpoints using val")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--width", type=int, default=448)
    train.add_argument("--height", type=int, default=256)
    train.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    train.add_argument("--no-pretrained", action="store_true", help="Use only for offline plumbing tests, not the baseline run")
    train.add_argument("--limit-train", type=int)
    train.add_argument("--limit-val", type=int)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--class-weighting", choices=["none", "balanced"], default="none", help="Optional capped inverse-square-root weights from resized training masks only")
    evaluate = commands.add_parser("evaluate", help="Report per-class metrics on an explicitly selected split")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    evaluate.add_argument("--limit", type=int)
    original = commands.add_parser("evaluate-original", help="Compare models against original-resolution labels, with small-anatomy and equal-case metrics")
    original.add_argument("--manifest", type=Path, required=True)
    original.add_argument("--checkpoint", type=Path, required=True)
    original.add_argument("--split", choices=["val", "test"], default="val")
    original.add_argument("--output", type=Path, required=True)
    original.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    original.add_argument("--limit", type=int)
    predict = commands.add_parser("predict", help="Export a trained model prediction, label raster, and provenance sidecar")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--media-id", required=True)
    predict.add_argument("--frame-number", type=int, required=True)
    predict.add_argument("--timestamp-ms", type=float, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    predict.add_argument("--threshold", type=float, default=0.5)
    predict.add_argument("--min-area", type=int, default=64)
    video = commands.add_parser("predict-video", help="Decode actual presentation timestamps and export a prediction for each video frame")
    video.add_argument("--checkpoint", type=Path, required=True)
    video.add_argument("--input", type=Path, required=True)
    video.add_argument("--media-id", required=True)
    video.add_argument("--output", type=Path, required=True)
    video.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    video.add_argument("--threshold", type=float, default=0.5)
    video.add_argument("--min-area", type=int, default=64)
    video.add_argument("--max-frames", type=int, help="Explicitly mark a partial export for a wiring check; omit for the full clip")
    compare = commands.add_parser("compare", help="Inspect original images, supplied annotations, and raw model predictions")
    compare.add_argument("--checkpoint", type=Path, required=True)
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--split", choices=["train", "val", "test"], default="val")
    compare.add_argument("--limit", type=int, default=3)
    compare.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    focused = commands.add_parser("compare-small", help="Show annotation-selected small-anatomy crops before and after a model change")
    focused.add_argument("--before-checkpoint", type=Path, required=True)
    focused.add_argument("--after-checkpoint", type=Path, required=True)
    focused.add_argument("--manifest", type=Path, required=True)
    focused.add_argument("--output", type=Path, required=True)
    focused.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")

    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            count = validate_export(load_json(args.input), args.schema)
            print(f"Valid: {count} frame result(s)")
        elif args.command == "export-unconfigured":
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
        elif args.command == "doctor":
            environment = {"python": sys.version.split()[0], "platform": platform.system(), "machine": platform.machine()}
            for package in ["torch", "torchvision", "numpy", "Pillow", "opencv-python-headless", "jsonschema"]:
                try:
                    environment[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    environment[package] = "not installed"
            try:
                import torch
                environment["devices"] = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
            except ImportError:
                environment["devices"] = "Install ml[train] to check accelerator availability"
            print(json.dumps(environment, indent=2))
        elif args.command == "download-endoscapes":
            from .download import download_seg50
            report = download_seg50(args.output_dir, metadata_only=args.metadata_only, progress=lambda text: print(text, flush=True))
            print(json.dumps({key: value for key, value in report.items() if key != "selectedEntries"}, indent=2))
        elif args.command == "prepare-endoscapes":
            from .dataset import prepare_dataset
            manifest = prepare_dataset(args.data_root, fps=args.fps,
                ignore_source_ids=args.ignore_source_id, exclude_frames=args.exclude_frame)
            _write_json(args.output, manifest)
            print(json.dumps(manifest["report"], indent=2))
        elif args.command == "preview":
            from .dataset import render_preview
            render_preview(load_json(args.manifest), args.output, split=args.split, limit=args.limit)
            print(f"Wrote dataset annotation preview: {args.output}")
        elif args.command == "train":
            from .training import train as train_model
            report = train_model(load_json(args.manifest), args.output_dir, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, width=args.width, height=args.height,
                device=args.device, pretrained=not args.no_pretrained, limit_train=args.limit_train,
                limit_val=args.limit_val, seed=args.seed, class_weighting=args.class_weighting)
            print(json.dumps({key: value for key, value in report.items() if key not in {"history", "config"}}, indent=2))
        elif args.command == "evaluate":
            from .training import evaluate as evaluate_model
            report = evaluate_model(args.checkpoint, load_json(args.manifest), split=args.split, device=args.device, limit=args.limit)
            _write_json(args.output, report)
            print(json.dumps(report, indent=2))
        elif args.command == "evaluate-original":
            from .evaluation import evaluate_original
            report = evaluate_original(args.checkpoint, load_json(args.manifest), split=args.split, device=args.device, limit=args.limit)
            _write_json(args.output, report)
            print(json.dumps({key: report[key] for key in ("split", "foreground_macro_iou", "small_anatomy_macro_iou", "small_anatomy_macro_dice")}, indent=2))
            print(f"Wrote original-resolution evaluation: {args.output}")
        elif args.command == "predict":
            from PIL import Image
            from .inference import SegmentationAdapter, export_prediction
            with Image.open(args.image) as image:
                width, height = image.size
            adapter = SegmentationAdapter(args.checkpoint, device=args.device, threshold=args.threshold, min_area=args.min_area)
            frame = FrameInput(args.media_id, args.frame_number, args.timestamp_ms, width, height, args.image)
            result = export_prediction(adapter, frame, args.output)
            print(f"Exported {len(result['structures'])} structures to {args.output}")
        elif args.command == "compare":
            from .diagnostics import render_comparison
            render_comparison(args.checkpoint, load_json(args.manifest), args.output,
                split=args.split, limit=args.limit, device=args.device)
            print(f"Wrote annotation/model comparison: {args.output}")
        elif args.command == "compare-small":
            from .comparison import render_small_anatomy_comparison
            render_small_anatomy_comparison(args.before_checkpoint, args.after_checkpoint,
                load_json(args.manifest), args.output, device=args.device)
            print(f"Wrote small-anatomy comparison: {args.output}")
        elif args.command == "predict-video":
            from .inference import SegmentationAdapter
            from .video import export_video
            adapter = SegmentationAdapter(args.checkpoint, device=args.device, threshold=args.threshold, min_area=args.min_area)
            report = export_video(adapter, args.input, args.output, args.media_id, max_frames=args.max_frames)
            print(json.dumps({key: value for key, value in report.items() if key not in {"frames", "training"}}, indent=2))
    except (ContractError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        if isinstance(error, ImportError):
            print("Install the pipeline dependencies: python -m pip install -e './ml[train]'", file=sys.stderr)
        return 1
    return 0


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
