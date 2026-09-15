# Inference latency

The historical first latency comparison holds the batch-002 checkpoint, FP32, 672 × 384
model input, thresholds and native image grids fixed. It compares these runtime
variants on the same validation frames:

| Variant | Auxiliary head | Schema validator |
| --- | --- | --- |
| baseline | computed, discarded | loaded and checked per call |
| cached_validation | computed, discarded | cached |
| no_auxiliary_head | disabled after strict loading | loaded and checked per call |
| combined | disabled after strict loading | cached |

The combined configuration is the adapter default. Training and checkpoint
loading still retain and strictly validate every auxiliary-head tensor. The
validation cache keeps at most eight validators, invalidates on file changes
or replacement, and preserves all per-result schema, finite-number and
cross-field checks. No schema or checkpoint was changed.

## Reproduce

Use the installed `ml[train]` dependencies and a fresh output directory:

```sh
.venv/bin/python -m holospex_ml benchmark-latency \
  --checkpoint ml/weights/current/best.pt \
  --manifest ml/outputs/endoscapes-manifest.json \
  --output-dir ml/outputs/latency-NEW-RUN \
  --device mps --repeats 3 --warmup 5 --profile-frames 10
```

Choose the actual runtime device explicitly (`mps`, `cpu`, or `cuda`); a missing
device fails instead of silently changing the comparison. CPU threads default
to four and are recorded. `--limit 2 --repeats 1 --warmup 1 --profile-frames 2`
is a smoke check only. Output directories are never overwritten.

The full run uses all 75 validation images, three repetitions, and four
variants: 225 measured calls per variant. It shuffles both image and variant
order with the recorded seed. Every native image size is warmed for each
variant. RGB images are decoded and resident before timing.

`benchmark.json` records hardware/software identity, checkpoint and manifest
hashes, every image identity/hash, source hashes and copies, raw timings,
per-repeat summaries, stage profiles and output parity. `RESULTS.md` summarizes
the result. An interrupted or invalid comparison remains explicitly failed.

## What the measurements mean

- **Headline latency:** decoded RGB to a validated `FrameResult`, with device
  synchronization only immediately before and after each complete call.
- **Diagnostic stages:** preprocessing, input transfer, model forward, native
  logit resizing/softmax, output transfer, geometry and validation. These run
  separately because extra synchronization alters execution.
- **Startup:** model loading/device placement and the first prediction, with
  imports already completed. This is not total application startup.
- **Parity:** every measured call must produce identical raw labels, frame
  JSON (including confidence and provenance), and withheld-component counts.
  A separate pass requires bitwise-identical main logits and outputs between
  baseline and combined on every selected image.

This benchmark excludes video/camera capture, file export, networking, browser
rendering and display. It does not establish live streaming throughput. The
live identification path is now implemented, but its capture-to-matching-overlay
delay must be measured separately from this offline adapter benchmark.

The provisional goal is p95 below 100 ms for the adapter. These changes do not
alter precision, input resolution, class mapping or learned weights. No training
or test-split evaluation is part of this comparison.

## September 14 result

The Mac MPS comparison was stopped at the user's request. The GPU process exited
with code 130; no related Python inference processes remained. Its partial
artifacts are preserved under `ml/outputs/latency-20260914-mps/`: 900 timed calls,
40 stage profiles, and 57 of 75 separate main-logit/output parity checks. The
report correctly records `failed` / `KeyboardInterrupt`; this is an intentionally
interrupted comparison, not an accepted completed result.

Optional `sysctl` hardware metadata collection is now best effort, so sandbox
denials no longer abort a benchmark. Before resuming, finish the parity/result
audit and generate the comparison chart in a new run. Targeted contract,
inference and synthetic benchmark tests passed during implementation. Those historical
measurements retain the old checkpoint; the current selection is documented in
[CURRENT_MODEL.md](CURRENT_MODEL.md). A new benchmark run
must use a fresh output directory; preserve the interrupted evidence.
