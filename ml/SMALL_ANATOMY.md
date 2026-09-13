# Small-anatomy experiments — September 12, 2026

Selected candidate: **`small-004-resolution/best.pt`**, epoch 9 from a
12-epoch budget at **672 × 384**. On the unchanged original validation masks,
four-class small-anatomy macro-IoU improves from **18.60% to 24.36%**, and
case-equal macro-IoU from **16.20% to 25.73%**, versus the three-epoch balanced
baseline. Results remain weak and uneven; this is not clinical validation.

## Why these controls

The first comparison increases the training budget from 3 to 12 epochs at
448 × 256. The second keeps that 12-epoch budget and increases input resolution
to 672 × 384. This distinguishes additional optimization from the resolution
change. All runs start from the same generic Torchvision COCO/VOC initialization,
use DeepLabV3–MobileNetV3, seed 42, MPS, batch 2, learning rate 0.0003, frozen
BatchNorm, and capped class weighting derived only from their resized train masks.
No decoder, sampling, augmentation, or loss-formula change was added.

The [train/validation mask audit](outputs/small-anatomy-resolution-audit.json)
found **no complete annotated instances lost** at 448 × 256 among the four
focus classes. Some tiny disconnected mask components disappear; none of the
lost focus-class components has an original area of 16 pixels or more. This
does not support blaming whole-instance disappearance for the weak predictions.
Resolution may still help boundaries and feature detail: median train artery
annotation area grows from 896 to 2,017 model-input pixels at 672 × 384.
The current main decoder has output stride 16, giving 28 × 16 versus 42 × 24
feature locations. 896 × 512 was audited but not trained in this comparison.

## Comparable validation scores

All candidates score the same **75 frames / 10 validation videos**, at the
original **854 × 480** annotation grid. Train uses 343 frames / 30 separate
videos. Source 255 remains ignored; quarantined `153_32700` with source ID 7
remains excluded. Evaluation scores **30,683,409 pixels**, ignoring **60,591**.
Logits resize to the original grid before argmax; no confidence or HUD polygon
filter is applied. These numbers differ slightly from training-resolution metrics.

“Pooled” aggregates pixels across frames. “Case-equal” first aggregates within
each video, then averages each class's defined video scores equally. Absent
truth with false positives scores zero; absent truth and prediction is undefined
and excluded. The six-class foreground score excludes background and checks
whether gains in the four focus classes sacrifice overall segmentation quality.

| Candidate | Budget / best epoch | Small pooled IoU / Dice | Small case-equal IoU / Dice | Six-class pooled IoU / Dice |
| --- | --- | ---: | ---: | ---: |
| Balanced baseline, 448 × 256 | 3 / 3 | 18.60% / 30.00% | 16.20% / 25.09% | 36.79% / 48.16% |
| Longer control, 448 × 256 | 12 / 9 | 22.43% / 36.08% | 19.73% / 29.90% | 40.64% / 53.06% |
| Resolution candidate, 672 × 384 | 12 / 9 | **24.36% / 38.28%** | **25.73% / 36.94%** | **42.94% / 55.17%** |

Within each run, `best.pt` was selected by six-class validation macro-IoU at
that run's training resolution. The table above compares those checkpoints on
the same original grid. Candidate selection used validation results only;
no new test evaluation or public-video result was used for this experiment.

| Focus class | Baseline pooled / case-equal IoU | Longer pooled / case-equal IoU | Resolution pooled / case-equal IoU | Validation cases with truth |
| --- | ---: | ---: | ---: | ---: |
| Cystic duct | 37.06% / 33.91% | 34.38% / 34.43% | 40.68% / 45.13% | 10 |
| Cystic artery | 11.48% / 12.07% | 14.49% / 11.77% | 16.06% / 14.98% | 8 |
| Cystic plate | 17.03% / 12.84% | 19.58% / 14.84% | 19.71% / 22.13% | 6 |
| Hepatocystic triangle dissection | 8.82% / 5.98% | 21.27% / 17.89% | 20.97% / 20.69% | 9 |

All four pooled and case-equal IoUs improve versus the three-epoch baseline.
The higher-resolution run does not improve every measure versus the longer
control: triangle pooled IoU decreases slightly, from 21.27% to 20.97%.
Artery precision improves from **13.61% to 23.21%** versus the baseline, while
recall falls from **42.37% to 34.27%**: fewer false positives accompany more
missed artery pixels. A higher mean IoU does not remove that tradeoff.

The [focused before/after sheet](outputs/small-004-resolution/comparison-small.png)
and [sidecar](outputs/small-004-resolution/comparison-small.json) select each
class's upper-median annotation area before inspecting model quality. Whole-frame
IoU improves for duct `126_13050` (**0.344 → 0.512**) and triangle `146_13275`
(**0.000 → 0.243**), but regresses for artery `126_11550` (**0.350 → 0.013**)
and plate `131_44875` (**0.392 → 0.253**). Crops are for display; metrics include
errors outside them. Supplied annotations are not reviewed clinical lesson answers.

## Artifacts and reproduction

Model ID: `holospex-deeplabv3-mobilenetv3`.
Selected version: **`2026-09-12T23:22:27.215217Z-epoch-9`**.
Selected [checkpoint](outputs/small-004-resolution/best.pt) SHA-256:
`448f77c258d054107f4cdde75a051b92c373503e7b0470d01bb4c27d27ede838`.

| Run | Saved configuration | Original-grid validation report |
| --- | --- | --- |
| `baseline-002-balanced` | [config](outputs/baseline-002-balanced/config.json) | [metrics](outputs/baseline-002-balanced/metrics-val-original.json) |
| `small-003-longer` | [config](outputs/small-003-longer/config.json) | [metrics](outputs/small-003-longer/metrics-val-original.json) |
| `small-004-resolution` | [config](outputs/small-004-resolution/config.json) | [metrics](outputs/small-004-resolution/metrics-val-original.json) |

Each report includes checkpoint and manifest hashes, preprocessing, per-video
metrics, and case counts. These artifacts are local and Git-ignored. The two
12-epoch histories sum to 219.8 and 410.2 seconds respectively; these recorded
epoch durations exclude setup, downloads, and handoff work and are not a throughput benchmark.

From the repository root, after the dataset/install steps in [TRAINING.md](TRAINING.md):
the following reproduce the settings in fresh directories, preserving recorded runs.

```sh
TORCH_HOME=ml/weights .venv/bin/python -m holospex_ml train --manifest ml/outputs/endoscapes-manifest.json --output-dir ml/outputs/reproduce-small-003 --device mps --epochs 12 --batch-size 2 --width 448 --height 256 --lr 0.0003 --class-weighting balanced
TORCH_HOME=ml/weights .venv/bin/python -m holospex_ml train --manifest ml/outputs/endoscapes-manifest.json --output-dir ml/outputs/reproduce-small-004 --device mps --epochs 12 --batch-size 2 --width 672 --height 384 --lr 0.0003 --class-weighting balanced
for run_dir in baseline-002-balanced reproduce-small-003 reproduce-small-004; do
  .venv/bin/python -m holospex_ml evaluate-original --manifest ml/outputs/endoscapes-manifest.json --checkpoint "ml/outputs/$run_dir/best.pt" --split val --device mps --output "ml/outputs/$run_dir/metrics-val-original.json"
done
.venv/bin/python -m holospex_ml compare-small --manifest ml/outputs/endoscapes-manifest.json --before-checkpoint ml/outputs/baseline-002-balanced/best.pt --after-checkpoint ml/outputs/reproduce-small-004/best.pt --device mps --output ml/outputs/reproduce-small-004/comparison-small.png
```

This is one seed, 30 training cases, and only 6–10 positive validation cases per
focus class. Correlated frames and repeated validation selection limit the evidence.
Next experiments should independently test case-balanced sampling or modest paired
augmentation against this control, retaining original-grid and per-case metrics.
Neither improvement is implemented here; further changes should not be chosen on test results.
