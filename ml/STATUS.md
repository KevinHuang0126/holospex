# ML run record — September 12, 2026

The current checkpoint is **`small-004-resolution/best.pt`**, selected at epoch 9
of a 12-epoch, 672 × 384 training run. On the same original-resolution validation
labels, small-anatomy mean IoU improves **18.60% → 24.36%**, equal-case small-anatomy
IoU improves **16.20% → 25.73%**, and six-class foreground IoU improves
**36.79% → 42.94%**. All four smaller classes improve aggregate IoU, but some
individual frames regress. See [SMALL_ANATOMY.md](SMALL_ANATOMY.md) for the
controlled comparison, precision/recall tradeoffs, commands, and exact artifacts.
This update uses validation only; there is no new test result for this checkpoint.

The initial Endoscapes pipeline completed two three-epoch runs.
Its earlier balanced-loss checkpoint reached **36.76% validation foreground
macro-IoU** and **35.57% test foreground macro-IoU / 46.01% macro-Dice**.
This is an early hackathon baseline; small-structure errors remain substantial.
It does not provide reviewed lesson answers, CVS decisions, or clinical validation.

## Data and preprocessing

- Official Endoscapes-Seg50: 493 original labeled frames, preserved unchanged.
- Usable split: **343 train / 75 validation / 74 test** frames from **30 / 10 / 10
  separate videos**. No video overlap across splits.
- Source label **255 is ignored** in training and metrics, never converted to
  background. Validation frame `153_32700` is quarantined because it contains
  unexplained source ID **7**; other unconfigured unknown IDs fail validation.
- Seven model channels: background, gallbladder, cystic duct, cystic artery,
  cystic plate, hepatocystic triangle dissection, tool. Source PNG IDs are
  remapped explicitly; they are not the same as the channel order.
- Images resize bilinearly to **448 × 256** for the initial runs and longer
  control, or **672 × 384** for the current checkpoint; masks use nearest neighbor.
  ImageNet normalization and frozen BatchNorm are shared by all runs.
- Background, gallbladder, and tool account for **97.01% of scored training
  pixels** at the initial 448 × 256 resolution. The smaller structures need
  separate inspection.

The [manifest](outputs/endoscapes-manifest.json) records split IDs, label policy,
excluded frame, source provenance, and the explicit 25-fps filename assumption.
That assumption is not a verified playback timebase for an arbitrary clip.
Dataset access, licensing, source references, and commands are in [TRAINING.md](TRAINING.md).

## Initial controlled comparison

Both runs use Torchvision `deeplabv3_mobilenet_v3_large`, generic
`COCO_WITH_VOC_LABELS_V1` initialization with new seven-channel output heads,
MPS on the M4 MacBook Air, seed 42, batch size 2, learning rate 0.0003,
and all available training/validation frames for three epochs. Class weighting
is the experiment change. Checkpoint selection uses validation foreground
macro-IoU; **epoch 3 was selected in each run**.

| Run | Training loss | Validation mIoU | Validation Dice | Test mIoU | Test Dice |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline-001` | Unweighted | 31.47% | 39.27% | Not evaluated | Not evaluated |
| `baseline-002-balanced` | Capped class weighting | **36.76%** | **48.13%** | **35.57%** | **46.01%** |

The balanced run improves validation mIoU by **5.29 percentage points**.
Weights use only resized, nonignored **training** pixels: inverse square-root
frequency, normalized by the mean foreground raw weight, then clipped to
`[0.1, 5.0]`. Exact counts/formula/float32 weights are saved in its
[config](outputs/baseline-002-balanced/config.json). Main and auxiliary training
losses are weighted; validation and test loss stay unweighted.

Metrics aggregate pixel confusion over each split at **448 × 256**, exclude
background from the six-class macro average, and exclude ignored pixels.
Validation scores 8,584,625 pixels and ignores 16,975; test scores 8,458,338 and
ignores 28,574. These are not original-resolution official benchmark results.
Test was evaluated for the selected balanced checkpoint after validation selection.

| Initial balanced model class | Validation IoU | Validation Dice | Test IoU | Test Dice |
| --- | ---: | ---: | ---: | ---: |
| Gallbladder | 75.76% | 86.21% | 73.62% | 84.80% |
| Cystic duct | 37.07% | 54.09% | 35.71% | 52.62% |
| Cystic artery | 11.47% | 20.58% | 26.52% | 41.92% |
| Cystic plate | 17.04% | 29.12% | 7.36% | 13.71% |
| Hepatocystic triangle dissection | 8.75% | 16.10% | 0.36% | 0.72% |
| Tool | 70.49% | 82.69% | 69.88% | 82.27% |

## Initial artifacts and identities

Shared model ID: `holospex-deeplabv3-mobilenetv3`.

| Run directory | Run ID (UTC) | Selected model version | Checkpoint |
| --- | --- | --- | --- |
| `baseline-001` | `2026-09-12T22:31:53.294408Z` | `2026-09-12T22:31:53.294408Z-epoch-3` | [best.pt](outputs/baseline-001/best.pt) |
| `baseline-002-balanced` | `2026-09-12T22:38:43.690799Z` | `2026-09-12T22:38:43.690799Z-epoch-3` | [best.pt](outputs/baseline-002-balanced/best.pt) |

Evidence: [baseline config](outputs/baseline-001/config.json),
[baseline validation](outputs/baseline-001/metrics-validation.json),
[balanced validation](outputs/baseline-002-balanced/metrics-validation.json),
[balanced test](outputs/baseline-002-balanced/metrics-test.json).
Each run also stores `history.json` and `last.pt`; use `best.pt` for this handoff.
Both configs record manifest SHA-256
`6c9197f8f17642514c282bf67d0764876d23a23cc5e1e5ad7593dd0e20b3db7b`
(canonical JSON metadata; this digest does not hash image/mask contents).
Run outputs, data, and weights are local Git-ignored artifacts.

## Observed failures and handoff boundaries

The [balanced validation comparison](outputs/baseline-002-balanced/comparison-val.png)
shows six deterministically selected cases; it was not ranked for good predictions.
The [sidecar](outputs/baseline-002-balanced/comparison-val.json) records source paths,
raw pixel counts, and separately filtered HUD summaries. Visual inspection confirms:

- **`141_40575`:** an extra purple cystic-plate region appears in the raw prediction.
  It contains 14,577 original-resolution pixels; the supplied annotation has zero.
- **`153_25950`:** gallbladder extends beyond the supplied annotated region:
  43,853 predicted pixels versus 13,148 annotated pixels (about 3.34× the area).

These are comparisons to supplied annotations, not clinical review. The sheet
uses raw argmax masks without confidence filtering. Exported softmax scores are
uncalibrated, and filtering does not establish correctness or resolve these failures.

- **Implemented and exercised:** data acquisition/audit, preprocessing, model
  training, validation/test evaluation, comparison sheets, and frame JSON/mask export.
- **Video export exercised:** all **120 consecutive frames** of the public
  [eight-second excerpt](outputs/demo-video-v1/zhou-2024-video2-excerpt.mp4) have
  direct predictions at original 1280 × 720 dimensions. First PTS is zero;
  final frame timestamp is 7927.52 ms. The [JSON array](outputs/demo-video-v1/predictions.json)
  passes both Python validation and the browser's `parseFrameResult` boundary.
  [Visual review](outputs/demo-video-v1/prediction-review.png),
  [export audit](outputs/demo-video-v1/predictions.info.json), and
  [media provenance](outputs/demo-video-v1/media-provenance.json) accompany it.
  No segmentation ground truth or accuracy score exists for this public clip.
  Updated predictions using the current checkpoint are in
  [demo-video-small-anatomy](outputs/demo-video-small-anatomy/predictions.json),
  paired with the same prepared MP4. All 120 updated frames pass both contract
  validators and preserve the earlier export's exact presentation timestamps.
  The earlier export remains available.
- **Not implemented:** live inference API, temporal tracking/smoothing, training
  augmentation, and reviewed clinical lesson content. Camera AR is a separate path.
- **Next handoff:** use [DEMO_MEDIA.md](DEMO_MEDIA.md) for the exact source,
  preparation command, media ID and attribution. Connect this clip/result array
  to the frontend and test playback alignment. The website still uses synthetic
  assets, so the complete integrated lesson remains unfinished.

## Checks and next experiment

All **91 ML tests pass**. They cover label conversion/ignore policy, leakage,
loss/metrics, safe checkpoint provenance, original-frame geometry, invalid
contours, real variable-PTS decoding, original-grid scoring, and annotation-selected
comparison crops. The original baseline completed validation and test evaluation;
the new candidates completed validation only and the selected one has a real clip export.

For the next training experiment, keep test data out of tuning: consider
case-balanced sampling or conservative paired augmentation, one change at a
time with the current model as control. Do not choose
settings by making the public demo overlays look more convincing. This test
split has now been inspected; further comparisons on it are not a fresh blind test.

On this Mac, importing both OpenCV and PyAV emits duplicate AVFoundation class
warnings from their bundled libraries. Both full clip exports completed, with
no observed crash. The pinned environment is an offline prototype; isolate
decoding in a fresh process before building a persistent live service.
