# ML pipeline — owner: ML lead

This package implements the offline anatomy pipeline: selective official
Endoscapes-Seg50 download, data validation and inspection, a seven-class
DeepLabV3-MobileNetV3 training baseline, evaluation, and original-frame exports.
Start with **[TRAINING.md](TRAINING.md)** for the data source, model choice,
commands and remaining gates from the Person 1 to-do list.
The actual runs and measured limitations are in **[STATUS.md](STATUS.md)**;
the public video and renderer handoff are in **[DEMO_MEDIA.md](DEMO_MEDIA.md)**.
The current small-anatomy improvements are documented in **[SMALL_ANATOMY.md](SMALL_ANATOMY.md)**.
For the prepared Google GPU workflow, see **[CLOUD_TRAINING.md](CLOUD_TRAINING.md)**.

Heavy libraries are optional: install `./ml[train]` for the full pipeline or
`./ml[data]` for data inspection. Plain `./ml` still provides lightweight
contract validation. Datasets and trained weights stay in ignored local
directories; they are not bundled in the repository. The retained unconfigured
adapter is a failure-handling test utility, not the real training path.

## Run from the repository root

Use Python 3.11 or newer. Install in a virtual environment so experiments do not
change system Python packages:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ./ml
.venv/bin/python -m holospex_ml validate assets/demo/frame-000.json
.venv/bin/python -m holospex_ml export-unconfigured ml/outputs/unconfigured.json --media-id demo --width 1280 --height 720
.venv/bin/python -m unittest discover -s ml/tests -v
```

For training, replace the installation command above with:

```sh
.venv/bin/python -m pip install -e './ml[train]'
.venv/bin/python -m holospex_ml doctor
```

`requirements-macos.txt` records the tested Python 3.14/macOS ARM64 dependency
versions. It is an environment snapshot, not a universal CUDA environment lock.

If dependencies are already installed, use `PYTHONPATH=ml/src python3 -m
holospex_ml ...` without an editable installation. `validate` accepts one frame
object or a JSON array of frame objects. Commands resolve the canonical
`contracts/schemas/frame-result.schema.json` from this checkout; use `--schema
/absolute/path/to/frame-result.schema.json` if distributing the Python package
separately. `export-unconfigured` validates before writing and refuses to replace
an existing file.

## Extending the implemented baseline

1. Use `predict-video` for decoded presentation timestamps and original image
   dimensions. Verify usable media, label mapping, and permission for each demo
   clip before depending on it. Keep source timebase/offset metadata with results.
2. `inference.SegmentationAdapter` implements `InferenceAdapter.predict` for
   our trained baseline. Load its checkpoint once, outside the per-frame loop.
   Add another model behind that interface only if results justify it.
3. The implemented model stretches input to its saved size, then resizes logits
   to the original frame before extracting geometry. A different adapter can
   use `ResizeTransform` for stretch or letterbox coordinates; preserve its
   exact dimensions and invert the transform before constructing frame results.
4. Map dataset/model classes explicitly to `contracts/anatomy.json`; never assume
   dataset label integers match another checkpoint. Label synthetic fixtures as
   `synthetic_mock`, reviewed annotations as `reviewed_annotation`, and actual
   inference as `ml_prediction`. Propagation uses its own source and prior-frame
   timestamp. Record actual model ID/version and per-instance confidence for
   predictions; confidence is a model score, not proof that a clinical criterion
   is met.
5. Validate every export against the shared schema and semantic checks before
   handing files to the web owner. Inspect overlays on original images, including
   image boundaries, timestamps, occlusion, and empty/error frames.

`ResizeTransform` works in continuous original-image pixel-edge coordinates:
`0 <= x <= width` and `0 <= y <= height`. It handles stretch and centered
letterbox transforms, including integer rounding. It does not implement crops,
rotation, mirroring, polygon clipping, or masks. If preprocessing adds one of
those operations, record and invert it explicitly; do not reuse an inaccurate
transform. Geometry in letterbox padding intentionally remains out of bounds so
the export validator catches it.

## Contract and integration rules

- JSON Schemas in `contracts/schemas/` are canonical. Coordinate, structure ID,
  status, or provenance changes require coordination with the web owner and an
  update to shared fixtures. Keep Python and browser semantic checks aligned.
- Return `ok` plus an empty `structures` array when inference ran and found no
  accepted detections. Use `unsupported`, `missing`, or `error` with a reason for
  unavailable or failed results. Every non-`ok` result must have no structures.
- Export original media timestamps in milliseconds, never the elapsed inference
  time. A result belongs to its media ID, timestamp, and original dimensions;
  the display owner controls when it is applicable and when overlays disappear.
- Keep lesson answers and content review outside ML prediction records. The model
  does not grade a learner or certify anatomy/operative safety.
- Surgical-image perception and physical-model marker tracking are separate
  adapters. The camera/AR owner implements marker registration and camera
  permission handling; this package does not identify anatomy through glasses.
- If a future display or live source is added, reuse the perception contract but
  revisit timing, registration, failure handling, and validation for that device.
  The current prototype is an educational demo.

## Data and evaluation

Keep datasets, patient media, checkpoints, virtual environments, and generated
exports out of Git. Use `ml/data/`, `ml/weights/`, and `ml/outputs/` locally; only
selected lightweight demo assets with documented use rights belong in `assets/demo/`.

Before training or reporting performance, create a reproducible split by
**video/case**, not by adjacent frames. Record dataset version, split IDs,
preprocessing, checkpoint identity, class mapping, threshold choices, and metrics
in an experiment note. Keep the demo case out of held-out evaluation claims if
it was used for model selection. Do not tune thresholds on the test set or label
reviewed ground truth as model predictions.

For this hack, prioritize a measured baseline and a verified export for the
selected lesson. An inference API and live streaming remain follow-on work;
the web app can consume precomputed results without a running Python server.
