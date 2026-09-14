# Current model

Use **`ml/weights/current/best.pt`** for the best verified local checkpoint.
Its [selection receipt](weights/current/selection.json) records source, checksum,
model version, validation metrics, previous selection and class tradeoffs.
Weights and runtime receipts remain outside Git; historical checkpoints remain.
Cloning or pulling `main` updates this handoff, but does not install the model
or replace predictions already loaded in a browser.

The September 14 batch-002 seed-42 checkpoint uses DeepLabV3–ResNet50 with surgical
MoCo initialization and balanced CE + 0.25 main Lovasz. Selected at epoch 34/53,
it uses 672 × 384 input and trained on 407 images. On the fixed 75-image native validation set:

| Metric | Previous best | Current |
| --- | ---: | ---: |
| Six-class foreground IoU | 52.0409% | 52.8251% |
| Small-anatomy pooled IoU | 34.6151% | 35.7053% |
| Equal-case small-anatomy IoU | 28.2247% | 32.2752% |

Checkpoint SHA-256:
`b406ed42ab0394edba22e1dde0edc2865a6346adb61c4bea7ab0bc00d08e1911`.
Model version: `2026-09-14T04:29:29.757933Z-epoch-34`.
The user explicitly authorized promotion, and strict CPU loading passed.
The winner improves aggregate IoU but triangle-dissection IoU declines 6.987
points; consult the [paired results](outputs/reviewed-batch002-20260914/reports/paired-final/RESULTS.md)
for precision/recall and case-level tradeoffs. Test remains excluded.

## Retrieve the checkpoint on a fresh checkout

Use an account with access to the project's private GCS bucket and the
`holospex` gcloud configuration described in [CLOUD_TRAINING.md](CLOUD_TRAINING.md).
The completed seed-42 Vertex job is `561146350624833536` in project
`576811516435`, region `us-central1`. Its exact completion record is bound to
the collected checkpoint and metrics:

```sh
.venv/bin/python ml/cloud/collect_vertex_results.py \
  --completion-uri gs://eastwest72hack26bos-501-holospex-ml/runs/review2-20260914-001-batch002-moco-lovasz-s42/attempts/20260914T042903Z-3691f7845f1b416da9e46380900f8a21/status/completed.json \
  --output-dir ml/outputs/current-model-download
```

The existing collector verifies every artifact's size and SHA-256 and refuses
an existing output directory. It retains training/evaluation records and cloud
receipts alongside `train/best.pt`. Confirm that the checkpoint matches this
selection before using it:

```sh
.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path

checkpoint = Path("ml/outputs/current-model-download/train/best.pt")
with checkpoint.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
if checkpoint.stat().st_size != 168351963 or digest != "b406ed42ab0394edba22e1dde0edc2865a6346adb61c4bea7ab0bc00d08e1911":
    raise SystemExit("Checkpoint does not match the current model selection")
print(f"Verified current checkpoint: {checkpoint}")
PY
```

Pass `--checkpoint ml/outputs/current-model-download/train/best.pt` directly,
or copy the verified checkpoint to `ml/weights/current/best.pt` for the commands
below. Links to the original selection receipt, paired report and video exports
refer to the ML lead's ignored local artifacts; those are not installed by Git
or by collecting this training run.

## Export predictions

From the repository root:

```sh
.venv/bin/python -m holospex_ml predict-video \
  --checkpoint ml/weights/current/best.pt \
  --input PATH_TO_PREPARED_VIDEO.mp4 \
  --media-id UNIQUE_CLIP_ID \
  --device auto --threshold 0.5 \
  --output ml/outputs/NEW_EXPORT/predictions.json
```

Choose a fresh output directory. The prepared video must have zero-start
presentation timestamps and no audio; see [TRAINING.md](TRAINING.md). The CLI
requires an explicit checkpoint argument, so use this stable path.

The browser consumes exported JSON, not weights. Open **Camera prototype →
Video**. Load the exact video first, then its JSON; select ML prediction,
Learn, threshold 0.5 and Show overlays.
The old Gupta export in `ml/outputs/gupta-cvs-lovasz/` retains run012 identity.
The [refreshed predictions.json](outputs/reviewed-batch002-20260914/gupta-current/predictions.json)
uses this checkpoint and covers all 1,056 frames of the exact
[Gupta clip](outputs/candidate-cvs-video/gupta-2023-cvs-anterior-posterior.mp4).
Decoded timestamps, original dimensions, raw masks, model/source checksums and
the actual frontend parser/matcher passed verification. At 10 seconds, frame
240 displays predictions without a matching-result warning. The browser footer
should show **1,056 validated frame results** after loading this JSON.
See [DEMO_MEDIA.md](DEMO_MEDIA.md) for clip provenance and attribution.

Endoscapes and SelfSupSurg retain their recorded CC BY-NC-SA 4.0 provenance.
Predictions remain separate from reviewed lesson answers.
