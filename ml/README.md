# ML pipeline — owner: ML lead

This package establishes the offline inference boundary for the two-day hack.
It includes a model adapter interface, original-image geometry conversion, and
export validation. **No model, training implementation, weights, clinical
annotations, or clinical performance claims are included.** The unconfigured
adapter returns `unsupported` and no geometry so teammates can handle that state
without mistaking it for a working detector.

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

If dependencies are already installed, use `PYTHONPATH=ml/src python3 -m
holospex_ml ...` without an editable installation. `validate` accepts one frame
object or a JSON array of frame objects. Commands resolve the canonical
`contracts/schemas/frame-result.schema.json` from this checkout; use `--schema
/absolute/path/to/frame-result.schema.json` if distributing the Python package
separately. `export-unconfigured` validates before writing and refuses to replace
an existing file.

## What to implement next

1. Write a decoder that preserves media timestamps and original image dimensions.
   Verify the exact dataset version, usable media, label map, and permission to
   use any selected demo content before depending on it.
2. Implement `InferenceAdapter.predict` in a new adapter module. Load a suitable
   checkpoint once, outside the per-frame loop. Keep framework dependencies
   optional and isolated in that adapter until a model is selected.
3. Record preprocessing using `ResizeTransform`. Resize and pad the actual image
   using its exact recorded dimensions. Decode model outputs, reverse that
   transform, clip masks/polygons to the original image, and remove degenerate
   shapes before creating frame results.
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

For this hack, prioritize a verified export for the selected lesson before
fine-tuning. Training, serving an inference API, and live streaming are follow-on
work once the offline rendering path is integrated.
