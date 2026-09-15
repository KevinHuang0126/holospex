# Current model

Use **`ml/weights/current/best.pt`**. On September 14, the user requested the
best verified model and an up-to-date `main` before cloud hosting. The selected
checkpoint is the **weight-decay-0.05 seed-42 DeepLabV3–ResNet50** from the
completed accuracy follow-ups, selected at epoch **43 / 53**.

The live runner and installer pin this model in
[`current_model.py`](src/holospex_ml/current_model.py).
The [tracked selection receipt](CURRENT_MODEL_SELECTION.json) records the ranked
33-run audit, exact artifact identity, retrieval receipt and class tradeoffs.
Weights remain outside Git. Pulling `main` does not install them or restart a host.

| Native validation metric | Previous batch-002 | Current |
| --- | ---: | ---: |
| Six-class foreground IoU | 52.8251% | 53.3060% |
| Small-anatomy pooled IoU | 35.7053% | 36.4221% |
| Equal-case small-anatomy IoU | 32.2752% | 32.8865% |

Scores use the same 75 validation images / ten cases at original 854 × 480
resolution, ignoring source 255 and excluding background only from the class
average. Primary improvement is **+0.4809 percentage points**. Artery and triangle
IoU improve 2.0770 and 3.1528 points; plate IoU falls 1.4733 points and plate
recall falls 7.5970 points. This is the best available audited checkpoint by
foreground IoU, not evidence of uniform class gains, test accuracy or clinical
validity. Retained epoch snapshots have not been searched at the native grid.

- Model ID: `holospex-deeplabv3-resnet50`
- Model version: `2026-09-14T16:39:12.476979Z-epoch-43`
- SHA-256: `9dc50d58fb2f605f5fc2a00652d7dae662f7584ab08e37502fb179b152873c1b`
- Size: **168,351,963 bytes**
- Input: 672 × 384; original checkpoint normalization and class mapping.
- Training: 407 images, surgical MoCo initialization, balanced CE + 0.25 Lovasz,
  AdamW weight decay 0.05, seed 42; complete 53-epoch schedule.
- Live confidence cutoff: **0.5**, unchanged and uncalibrated.

## Retrieve and verify

Use an account with access to the private GCS bucket and the `holospex` gcloud
configuration in [CLOUD_TRAINING.md](CLOUD_TRAINING.md). The source Vertex job is
`9068226144402145280` in `us-central1`, project `576811516435`.
The collection command verifies every artifact size and SHA-256 and requires
a fresh output directory:

```sh
.venv/bin/python ml/cloud/collect_vertex_results.py \
  --completion-uri gs://eastwest72hack26bos-501-holospex-ml/runs/followup-20260914-1509-001-moco-wd005/attempts/20260914T163844Z-80074ab281774218925909553568b1f1/status/completed.json \
  --output-dir ml/outputs/current-model-download-wd005

.venv/bin/python ml/cloud/audit_autonomous_results.py \
  ml/outputs/current-model-download-wd005
```

For a fresh installation:

```sh
npm run model:prepare -- \
  --checkpoint ml/outputs/current-model-download-wd005/train/best.pt
npm run model:serve
```

The installer refuses to overwrite different existing weights. For an upgrade,
first stop the old runner and move its checkpoint and any `selection.json` into
a uniquely named directory under `ml/weights/previous/`; then run the commands
above. Preserve the old files for rollback. On the ML lead's checkout, the old
checkpoint and receipt are already backed up under
`ml/weights/previous/b406ed42ab0394edba22e1dde0edc2865a6346adb61c4bea7ab0bc00d08e1911/`.

A supplied checkpoint must pass the pinned checksum and metadata checks before
readiness. `GET /api/identify` must advertise the version above after the model
host is updated and restarted. The Vercel deployment alone does not install
weights or run Python; see [hosting](../docs/live-feed.md).

## Video exports and compatibility

The existing 1,056-frame Gupta export at
`ml/outputs/reviewed-batch002-20260914/gupta-current/predictions.json` uses the
**previous** batch-002 model (`2026-09-14T04:29:29.757933Z-epoch-34`).
It is preserved and must not be relabeled as this model's output. Use the
live identification path with the selected runner, or explicitly generate a
fresh export for offline playback:

```sh
.venv/bin/python -m holospex_ml predict-video \
  --checkpoint ml/weights/current/best.pt \
  --input PATH_TO_PREPARED_VIDEO.mp4 \
  --media-id UNIQUE_CLIP_ID --device cpu --threshold 0.5 \
  --output ml/outputs/NEW_EXPORT/predictions.json
```

Use the exact zero-start video and its exported frame identities; see
[DEMO_MEDIA.md](DEMO_MEDIA.md). Existing remote deployments and already-loaded
JSON retain their old identity until explicitly updated. No new cloud job,
full-video export, test evaluation or hosting deployment accompanies this
selection. Endoscapes and SelfSupSurg retain their recorded CC BY-NC-SA 4.0
provenance; predictions remain separate from reviewed lesson answers.
