"""Matched batch-one latency measurements; never train or select checkpoints.

Wall timings use the ordinary adapter with synchronization only at the outer
boundaries. Instrumented stage timings and main-logit parity are separate passes.
The input set is validation only; decoded RGB is resident before timing starts.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from .adapters import FrameInput
from . import inference, validation
from .model import resolve_device


VARIANTS = {
    "baseline": {"auxiliary_head": True, "cached_validation": False},
    "cached_validation": {"auxiliary_head": True, "cached_validation": True},
    "no_auxiliary_head": {"auxiliary_head": False, "cached_validation": False},
    "combined": {"auxiliary_head": False, "cached_validation": True},
}


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Latency samples must be nonempty, finite and nonnegative")
    return {"count": len(values), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95, method="linear")),
            "min_ms": float(values.min()), "max_ms": float(values.max())}


def _output_digest(output):
    result, labels, withheld = output
    payload = json.dumps([result, withheld], sort_keys=True, allow_nan=False, separators=(",", ":"))
    return {"frame_result_and_withheld_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "labels_shape": list(labels.shape), "labels_dtype": str(labels.dtype)}


def _atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def _variant(adapter, auxiliary, name):
    """Restore runtime state even on error; only used by this serial benchmark."""
    config = VARIANTS[name]
    previous = adapter.model.aux_classifier
    adapter.model.aux_classifier = auxiliary if config["auxiliary_head"] else None
    validate = partial(validation.validate_frame_result, use_cache=config["cached_validation"])
    try:
        with patch.object(inference, "validate_frame_result", validate):
            yield
    finally:
        adapter.model.aux_classifier = previous


def _load_samples(manifest, limit=None):
    samples = sorted((s for s in manifest["samples"] if s["split"] == "val"),
                     key=lambda s: (str(s["videoId"]), s["frameNumber"]))
    if not samples:
        raise ValueError("Manifest contains no validation samples")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:limit]
    loaded, records, identities = [], [], set()
    for sample in samples:
        identity = (str(sample["videoId"]), sample["frameNumber"])
        if identity in identities:
            raise ValueError(f"Duplicate validation frame: {identity}")
        identities.add(identity)
        path = Path(sample["imagePath"]).resolve()
        image_hash = _sha256(path)
        started = time.perf_counter()
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"))
        decode_ms = (time.perf_counter() - started) * 1000
        frame = FrameInput(f"endoscapes-{sample['videoId']}", sample["frameNumber"],
                           sample["timestampMs"], rgb.shape[1], rgb.shape[0], path)
        loaded.append((frame, rgb))
        records.append({"image_path": str(path), "image_sha256": image_hash,
                        "video_id": str(sample["videoId"]), "frame_number": frame.frame_number,
                        "timestamp_ms": frame.timestamp_ms, "width": frame.width, "height": frame.height,
                        "setup_file_decode_ms": decode_ms})
    return loaded, records


def _write_markdown(report, output_dir):
    lines = ["# Local inference latency", "", f"Device: **{report['device']}**; batch size 1; "
             f"{len(report['samples'])} validation images, {report['protocol']['repeats']} repeats.", "",
             "Decoded RGB to validated FrameResult; excludes image/video decode, loading, export, capture and display.", "",
             "| Variant | Calls | Median (ms) | p95 (ms) |", "| --- | ---: | ---: | ---: |"]
    for name, summary in report["summary"].items():
        lines.append(f"| {name} | {summary['count']} | {summary['median_ms']:.2f} | {summary['p95_ms']:.2f} |")
    lines.extend(["", f"Combined median reduction: **{report['combined_median_reduction_percent']:.2f}%**.",
                  f"Combined p95 below 100 ms: **{report['combined_p95_below_100ms']}**.", "",
                  f"Exact output parity in timed calls: **{report['parity']['timed_outputs_equal']}**.",
                  f"Exact main-logit and output parity across {len(report['logit_parity'])} images: "
                  f"**{report['parity']['main_logits_and_outputs_equal']}**.", "",
                  "## Separately instrumented stages", "",
                  "Extra stage-boundary synchronization changes execution; these are diagnostic timings, not the headline latency.", "",
                  "| Stage | Baseline median (ms) | Combined median (ms) |", "| --- | ---: | ---: |"])
    for stage in report["stage_summary"]["baseline"]:
        lines.append(f"| {stage} | {report['stage_summary']['baseline'][stage]['median_ms']:.2f} | "
                     f"{report['stage_summary']['combined'][stage]['median_ms']:.2f} |")
    lines.extend(["", "## Protocol and limits", "",
                  f"Checkpoint SHA-256: `{report['checkpoint_sha256']}`.",
                  f"Manifest SHA-256: `{report['manifest_sha256']}`.",
                  f"Model load/device placement: {report['model_load_ms']:.2f} ms (fresh model in an already imported process).", "",
                  "- Variant order is shuffled within each frame and repeat using a recorded seed.",
                  "- Warmups, diagnostic profiles, parity passes, hashing and JSON writes are excluded from headline timings.",
                  "- Raw labels, geometry, confidence, provenance and withheld-component counts must match exactly.",
                  "- Main-logit parity is checked separately between baseline and combined on every selected image.",
                  "- This is local offline adapter latency, not live video throughput or capture-to-display delay.",
                  "- No training, test-split evaluation, precision change or input-resolution change.",
                  "- Image hashes, runtime/source identities, per-call values and per-repeat summaries are in benchmark.json.", ""])
    (output_dir / "RESULTS.md").write_text("\n".join(lines))


def benchmark_latency(checkpoint_path, manifest_path, output_dir, *, device,
                      repeats=3, warmup=5, profile_frames=10, limit=None, seed=42, cpu_threads=4):
    """Run four matched runtime variants and refuse to overwrite any prior run."""
    if min(repeats, warmup, profile_frames, cpu_threads) < 1:
        raise ValueError("repeats, warmup, profile_frames and cpu_threads must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace benchmark directory {output_dir}")
    resolved = resolve_device(device)
    torch.set_num_threads(cpu_threads)
    checkpoint_path, manifest_path = Path(checkpoint_path).resolve(), Path(manifest_path).resolve()
    manifest = validation.load_json(manifest_path)
    loaded, samples = _load_samples(manifest, limit)
    shape_representatives = {}
    for index, (frame, _) in enumerate(loaded):
        shape_representatives.setdefault((frame.width, frame.height), index)
    warmup_indices = list(shape_representatives.values())
    while len(warmup_indices) < warmup:
        warmup_indices.append(len(warmup_indices) % len(loaded))
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = {str(p): _sha256(p) for p in [Path(__file__), Path(inference.__file__),
               Path(validation.__file__), Path(__file__).with_name("model.py"), validation.default_schema_path()]}
    report = {"format_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
              "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
              "device": str(resolved), "platform": platform.platform(), "machine": platform.machine(),
              "python": platform.python_version(), "torch": torch.__version__,
              "torchvision": importlib.metadata.version("torchvision"), "cpu_threads": torch.get_num_threads(),
              "source_sha256": sources, "samples": samples, "variants": VARIANTS,
              "protocol": {"batch_size": 1, "repeats": repeats, "warmup_calls_per_variant": len(warmup_indices),
                           "warmup_sample_indices": warmup_indices,
                           "native_sizes": [list(size) for size in shape_representatives],
                           "profile_frames": min(profile_frames, len(loaded)), "seed": seed,
                           "percentile_method": "numpy linear", "input": "predecoded resident RGB",
                           "headline_sync": "device synchronization only immediately before/after complete adapter call",
                           "scope": "decoded RGB to valid FrameResult; startup/decode/exports/capture/display excluded"},
              "calls": [], "profiles": [], "logit_parity": []}
    if platform.system() == "Darwin":
        report["hardware"] = {}
        for key in ("hw.model", "machdep.cpu.brand_string", "hw.memsize"):
            try:
                value = subprocess.run(["/usr/sbin/sysctl", "-n", key],
                                       capture_output=True, text=True, check=True,
                                       timeout=5).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                value = None  # Optional metadata must not prevent a benchmark.
            report["hardware"][key] = value
    report["package_versions"] = {name: importlib.metadata.version(name) for name in
                                  ("numpy", "Pillow", "opencv-python-headless", "jsonschema")}
    source_dir = output_dir / "source"
    source_dir.mkdir()
    for path in sources:
        (source_dir / Path(path).name).write_bytes(Path(path).read_bytes())
    report_path = output_dir / "benchmark.json"
    _atomic_json(report_path, report)
    try:
        started = time.perf_counter()
        adapter = inference.SegmentationAdapter(checkpoint_path, device=resolved, run_auxiliary_head=True)
        _synchronize(resolved)
        report["model_load_ms"] = (time.perf_counter() - started) * 1000
        report["model"] = {"id": adapter.checkpoint["model_id"], "version": adapter.checkpoint["model_version"],
                           "input_size": adapter.checkpoint["input_size"], "threshold": adapter.threshold,
                           "min_area": adapter.min_area, "dtype": str(next(adapter.model.parameters()).dtype)}
        if adapter.checkpoint["classes"] != manifest["classes"]:
            raise ValueError("Checkpoint and manifest class mappings do not match")
        auxiliary = adapter.model.aux_classifier
        if auxiliary is None:
            raise ValueError("Benchmark requires a checkpoint with an auxiliary head")
        for name in VARIANTS:
            with _variant(adapter, auxiliary, name):
                for position, index in enumerate(warmup_indices):
                    started = time.perf_counter()
                    adapter.predict_rgb_details(*loaded[index])
                    if name == "baseline" and position == 0:
                        _synchronize(resolved)
                        report["first_prediction_ms"] = (time.perf_counter() - started) * 1000
                _synchronize(resolved)
        print(f"Warmup complete: {resolved}, {len(loaded)} images, {repeats} repeats, four variants", flush=True)
        references, mismatches = {}, []
        rng = random.Random(seed)
        for repeat in range(repeats):
            order = list(range(len(loaded)))
            rng.shuffle(order)
            for position, index in enumerate(order):
                variants = list(VARIANTS)
                rng.shuffle(variants)
                for name in variants:
                    with _variant(adapter, auxiliary, name):
                        _synchronize(resolved)
                        started = time.perf_counter()
                        output = adapter.predict_rgb_details(*loaded[index])
                        _synchronize(resolved)
                        elapsed_ms = (time.perf_counter() - started) * 1000
                    fingerprint = _output_digest(output)
                    references.setdefault(index, fingerprint)
                    equal = fingerprint == references[index]
                    if not equal:
                        mismatches.append({"repeat": repeat, "sample_index": index, "variant": name})
                    report["calls"].append({"repeat": repeat, "sample_index": index, "variant": name,
                                            "latency_ms": elapsed_ms, "structure_count": len(output[0]["structures"]),
                                            "output_equal": equal, **fingerprint})
                if (position + 1) % 25 == 0 or position + 1 == len(order):
                    print(f"Timed repeat {repeat + 1}/{repeats}: {position + 1}/{len(order)} frames", flush=True)
                    _atomic_json(report_path, report)
        for index in np.linspace(0, len(loaded) - 1, min(profile_frames, len(loaded)), dtype=int):
            for name in VARIANTS:
                timings = {}
                with _variant(adapter, auxiliary, name):
                    adapter.predict_rgb_details(*loaded[index], timings=timings)
                report["profiles"].append({"sample_index": int(index), "variant": name, **timings})
        print("Stage profiles complete; checking main logits and outputs on every selected image", flush=True)
        for index, sample in enumerate(loaded):
            fingerprints = []
            for name in ("baseline", "combined"):
                captured = {}

                def capture(_module, _inputs, outputs):
                    # This copy is ONLY in the separate parity pass, never in timed calls.
                    logits = outputs["out"].detach().cpu().contiguous().numpy()
                    captured["logits_sha256"] = hashlib.sha256(logits.tobytes()).hexdigest()
                    captured["logits_shape"] = list(logits.shape)
                    captured["logits_dtype"] = str(logits.dtype)

                handle = adapter.model.register_forward_hook(capture)
                try:
                    with _variant(adapter, auxiliary, name):
                        output = adapter.predict_rgb_details(*sample)
                    fingerprints.append({**captured, **_output_digest(output)})
                finally:
                    handle.remove()
            report["logit_parity"].append({"sample_index": index, "equal": fingerprints[0] == fingerprints[1],
                                           "baseline": fingerprints[0], "combined": fingerprints[1]})
            if (index + 1) % 25 == 0 or index + 1 == len(loaded):
                print(f"Parity: {index + 1}/{len(loaded)} images", flush=True)
                _atomic_json(report_path, report)
        report["parity"] = {"timed_outputs_equal": not mismatches, "mismatches": mismatches,
                            "main_logits_and_outputs_equal": all(p["equal"] for p in report["logit_parity"])}
        report["summary"] = {name: _summary([c["latency_ms"] for c in report["calls"] if c["variant"] == name])
                             for name in VARIANTS}
        report["repeat_summary"] = [{name: _summary([c["latency_ms"] for c in report["calls"]
                                     if c["variant"] == name and c["repeat"] == repeat]) for name in VARIANTS}
                                    for repeat in range(repeats)]
        stages = [key for key in report["profiles"][0] if key.endswith("_ms")]
        report["stage_summary"] = {name: {stage: _summary([p[stage] for p in report["profiles"]
                                           if p["variant"] == name]) for stage in stages} for name in VARIANTS}
        report["combined_median_reduction_percent"] = 100 * (1 - report["summary"]["combined"]["median_ms"] /
                                                                 report["summary"]["baseline"]["median_ms"])
        report["combined_p95_below_100ms"] = report["summary"]["combined"]["p95_ms"] < 100
        if not report["parity"]["timed_outputs_equal"] or not report["parity"]["main_logits_and_outputs_equal"]:
            raise RuntimeError("Output parity failed; see benchmark.json; latency improvement must not be accepted")
        if any(_sha256(path) != digest for path, digest in sources.items()):
            raise RuntimeError("Benchmark source/schema changed during measurement")
        if _sha256(checkpoint_path) != report["checkpoint_sha256"] or _sha256(manifest_path) != report["manifest_sha256"]:
            raise RuntimeError("Checkpoint or manifest changed during measurement")
        report["status"] = "completed"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(report_path, report)
        _write_markdown(report, output_dir)
        return report
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        _atomic_json(report_path, report)
        raise
