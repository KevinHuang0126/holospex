# ML run record — September 14, 2026

The [second reviewed-data training batch](REVIEW_BATCH_002.md) is **complete**.
Both approved A100 jobs succeeded, completing **53 epochs each / 106 total**.
All 36 collected artifacts passed checksum verification; independent audits
recomputed the unchanged 75-image native validation metrics. The controller
closed at **05:11:20 UTC**, ahead of its 06:25:44 deadline, with no owned jobs
remaining active.

The best local checkpoint is now **`ml/weights/current/best.pt`**, batch-002 seed 42,
selected at epoch 34/53. Native six-class foreground IoU improved from
**52.0409% to 52.8251% (+0.7842 percentage points)**. Small-anatomy pooled IoU
improved **34.6151% → 35.7053%**, and equal-case small IoU **28.2247% → 32.2752%**.
The other new seed scored 51.9172%, improving its matched 51.6340% control.
Across the two paired seeds, mean gains were +0.5338 foreground, +0.7285 small
pooled, and +2.7196 equal-case percentage points.

The winner improves duct, artery and plate IoU, but triangle-dissection IoU
falls 6.987 points, with more false positives; some validation cases regress.
These tradeoffs are recorded in the [paired report](outputs/reviewed-batch002-20260914/reports/paired-final/RESULTS.md).
No test inference or clinical validation was performed. The full local and
extracted-source suites pass **323 tests**. Strict CPU loading of the promoted
checkpoint passed. The [refreshed Gupta predictions](outputs/reviewed-batch002-20260914/gupta-current/predictions.json)
now cover all 1,056 frames and pass decoded-frame identity checks and the actual
frontend parser/matcher, including the 10-second frame.

The new review contributes 119 eligible masks on 46 images from 25 TRAIN cases
(after two explicit lead exclusions), giving 407 train / 75 val / 74 test images.
Original reviews, inherited data, historical checkpoints and old prediction
exports are preserved. See [current model usage](CURRENT_MODEL.md), the
[selection receipt](weights/current/selection.json), and [experiment log](EXPERIMENT_LOG.md).

## Historical September 13 results

All sections below preserve the September 13 run record. References to the
selected demo, pending integration and test counts describe that earlier stage;
use the September 14 summary above and [CURRENT_MODEL.md](CURRENT_MODEL.md)
for the current checkpoint and video handoff.

The [four-hour autonomous search](AUTONOMOUS_TRAINING.md) is **complete**. The
controller stopped at its 23:44:23 UTC deadline; all **15 owned Vertex jobs
succeeded**, completing **900 full epochs**, with every run independently
audited and no active jobs remaining. Run012, MoCo-initialized DeepLabV3–ResNet50
with main-head Lovasz, leads at **52.0409% six-class foreground IoU** on the
unchanged 75-frame native validation set, up **6.1692 percentage points** from
the 45.8717% reviewed reference. It selected epoch 34/60 and reached **34.6151%
pooled small-anatomy IoU** and **28.2247% equal-case small-anatomy IoU**. The
**75% target was not reached**. Its three-seed foreground mean is **51.5111%**
(range 50.8585–52.0409%). See the [final results](outputs/autonomous-20260913/reports/final/RESULTS.md),
[seed comparison](outputs/autonomous-20260913/reports/replication-results.md),
and [experiment log](EXPERIMENT_LOG.md). The local ML suite passes **312 tests**.
The research leader was not promoted at the close of this search, which did not
evaluate the test split.

The selected demo checkpoint at that stage was **`small-004-resolution/best.pt`**, selected at epoch 9
of a 12-epoch, 672 × 384 training run. On the same original-resolution validation
labels, small-anatomy mean IoU improves **18.60% → 24.36%**, equal-case small-anatomy
IoU improves **16.20% → 25.73%**, and six-class foreground IoU improves
**36.79% → 42.94%**. All four smaller classes improve aggregate IoU, but some
individual frames regress. See [SMALL_ANATOMY.md](SMALL_ANATOMY.md) for the
controlled comparison, precision/recall tradeoffs, commands, and exact artifacts.
This update uses validation only; there is no new test result for this checkpoint.
All five cloud experiments are complete. The selected demo checkpoint remains
unchanged: the best pooled cloud result improves small-anatomy IoU to 26.36%,
but lowers case-equal small IoU to 25.35% and artery recall from 34.27% to 24.63%.
See [CLOUD_RESULTS.md](CLOUD_RESULTS.md) for every candidate and the decision.

The separate [Dice iteration](DICE_ITERATION.md) is also complete: six fresh
40-epoch A100 jobs compared balanced CE against CE + foreground generalized Dice
at seeds 42, 43, and 44, retaining 672 × 384 input and the original split.
All six jobs succeeded and their artifacts were checksum-verified. Mean small
IoU fell **25.13% → 24.60%**, case-equal small IoU **22.42% → 21.49%**, and plate
recall **24.12% → 15.63%**. Artery recall rose **21.06% → 22.88%**. This variant
does not improve performance overall; the selected demo remains unchanged.
See [DICE_RESULTS.md](DICE_RESULTS.md) for all seeds, per-class tradeoffs,
independent verification and next steps.
The Dice packaged source passed 135 tests; its final workspace passed 143 ML tests.
Additional source data was acquired separately and did not enter these six runs.
Acquisition is complete: [852 usable new Endoscapes TRAIN images with 3,951 boxes](DATA_EXPANSION.md)
and [8,080 CholecSeg8k image/mask pairs](ADDITIONAL_MASK_DATA.md) are audited and
copied to private GCP storage. The latter has only partial Holospex class coverage.

The [teammate mask-review pilot](MASK_REVIEW.md) is now generated and packaged:
**20 images from 20 new TRAIN cases, with 52 SAM 2.1 box-prompted proposals**
(16 arteries, 16 ducts, 9 plates, 11 triangle-dissection regions). Vertex job
`4112639277085491200` succeeded on one Spot A100; all 83 output artifacts and
source/mask identities were checksum-verified. The portable HTML editor supports
explicit named reviews, brush/erase corrections, JSON export/resume and strict
Python import. The frozen reviewed-data training source passes **192 ML tests**;
the review UI previously passed **12 core tests**. The [first returned anatomy review](REVIEW_PILOT_RESULTS.md) now
contains all 52 decisions: 36 accepted unchanged, 13 edited and approved,
2 needs-expert and 1 rejected. All 49 approved masks were validated and imported
as separate partial masks. The lead confirmed both flagged edits were finished
and approved ignoring overlapping pixels. The [derived training targets](PARTIAL_TRAINING.md)
now retain 323,094 reviewed pixels and add 18 images: **361 train / 75 validation
/ 74 test**, with original samples and holdouts unchanged. Native target hashes,
actual loader resizing and ignored-pixel CE/Dice gradients pass verification.
The [reviewed-data training comparison](REVIEWED_DATA_RESULTS.md) completed
six fresh A100 jobs: three original-data controls at 42 epochs and
three expanded-data candidates at 40 epochs, paired by seeds 42/43/44. This
matches optimizer-update budgets within 0.22% at the unchanged 672 × 384 input
and balanced-CE settings. All three pairs improve both small-anatomy measures:
mean pooled IoU **24.40% → 27.23%** and equal-case IoU **22.59% → 24.65%**.
Artery and duct show the clearest gains; plate has only +0.33 pooled IoU points
and loses 1.10 equal-case points, with lower precision. This supports another
bounded labeling batch with particular attention to plate boundaries. All 93
artifacts were checksum-verified; the comparison independently recomputes
metrics and binds actual training file hashes to the reviewed package. The
final workspace passes **205 ML tests**. All jobs have finished, test was not
evaluated, and the selected demo checkpoint remains unchanged.

The [higher-resolution iteration](RESOLUTION_ITERATION.md) is complete. Two
matched 40-epoch A100 runs tested 896 × 512 and 1120 × 640 against the existing
672 × 384 cloud control. The 896 × 512 run regressed small-anatomy results. The
1120 × 640 run improved pooled small IoU **26.36% → 27.20%** and artery recall
**24.63% → 43.87%**, but equal-case small IoU fell **25.35% → 24.76%**, artery
precision fell **30.39% → 17.59%**, and plate IoU fell **18.28% → 14.03%**.
Its median local CPU adapter latency was 227.07 ms versus 121.89 ms for the
control. Both jobs succeeded and all artifacts were verified; neither used the
test split or unreviewed masks. The selected demo checkpoint remains unchanged;
those resolution jobs have finished.

The initial Endoscapes pipeline completed two three-epoch runs.
Its earlier balanced-loss checkpoint reached **36.76% validation foreground
macro-IoU** and **35.57% test foreground macro-IoU / 46.01% macro-Dice**.
This is an early hackathon baseline; small-structure errors remain substantial.
It does not provide reviewed lesson answers, CVS decisions, or clinical validation.

### Cloud training and implemented experiment options

All five Vertex jobs reached **JOB_STATE_SUCCEEDED** on September 13, 2026,
using NVIDIA A100 GPUs. Each completed training and original-grid validation;
all final artifacts were downloaded and checksum-verified. No training job in
this batch remains active. See [CLOUD_EXPERIMENTS.md](CLOUD_EXPERIMENTS.md) for
the frozen plan, [CLOUD_RESULTS.md](CLOUD_RESULTS.md) for comparisons, and
[CLOUD_TRAINING.md](CLOUD_TRAINING.md) for reproduction and artifact retrieval.

| Run | Vertex custom job ID | Submitted (September 13, UTC) |
| --- | --- | --- |
| 12-epoch matched control | `3017512501681061888` | 03:23:17 |
| 40-epoch longer MobileNet control | `6848949884666511360` | 03:38:23 |
| 40-epoch ResNet50 candidate | `7671982716568469504` | 03:38:24 |
| 40-epoch augmentation candidate | `2909989060577591296` | 03:43:45 |
| 40-epoch case-balanced candidate | `6893422930986795008` | 03:54:06 |

The training package now supports DeepLabV3–ResNet50 in addition to the existing
MobileNetV3 model, optional mild paired augmentation, and optional case-balanced
sampling. Augmentation and replacement sampling apply only to training;
class weights still count each unaugmented training mask once. Defaults remain
MobileNetV3, no augmentation, and uniform sampling. Existing checkpoints and the
segmentation output contract remain compatible. Each change is isolated in the
[frozen five-run plan](CLOUD_EXPERIMENTS.md). None of the alternatives resolves
the case-equal/artery-recall tradeoff enough to replace the local demo checkpoint.

Cloud runs preserve coherent completed-epoch artifacts in private Cloud Storage
and evaluate original-resolution validation labels. Exact resume is not
implemented: optimizer, RNG, and sampler state are not saved. Interrupted
attempts retain distinct identities. No cloud test-set results are claimed here.

### Data and preprocessing

- Official Endoscapes-Seg50: 493 original labeled frames, preserved unchanged.
- Usable split: **343 train / 75 validation / 74 test** frames from **30 / 10 / 10
  separate videos**. No video overlap across splits.
- Source label **255 is ignored** in training and metrics, never converted to
  background. Validation frame `153_32700` is quarantined because it contains
  unexplained source ID **7**; other unconfigured unknown IDs fail validation.
- Seven model channels: background, gallbladder, cystic duct, cystic artery,
  cystic plate, hepatocystic triangle dissection, tool. Source PNG IDs are
  remapped explicitly; they are not the same as the channel order.
- Images resize bilinearly to **448 × 256** for the earlier local controls,
  **672 × 384** for the then-selected checkpoint and original cloud batch, and the
  completed resolution candidates used **896 × 512** and **1120 × 640**; masks use nearest neighbor.
  ImageNet normalization and frozen BatchNorm are shared by all runs.
- Background, gallbladder, and tool account for **97.01% of scored training
  pixels** at the initial 448 × 256 resolution. The smaller structures need
  separate inspection.

The [manifest](outputs/endoscapes-manifest.json) records split IDs, label policy,
excluded frame, source provenance, and the explicit 25-fps filename assumption.
That assumption is not a verified playback timebase for an arbitrary clip.
Dataset access, licensing, source references, and commands are in [TRAINING.md](TRAINING.md).

### Initial controlled comparison

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

### Initial artifacts and identities

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

### Observed failures and handoff boundaries

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
- **Implemented experiment options:** ResNet50 checkpoint dispatch, mild paired
  training augmentation, and case-balanced training sampling, covered by CPU
  tests and completed CUDA runs. Their validation comparisons are recorded above.
- **Video export exercised:** all **120 consecutive frames** of the public
  [eight-second excerpt](outputs/demo-video-v1/zhou-2024-video2-excerpt.mp4) have
  direct predictions at original 1280 × 720 dimensions. First PTS is zero;
  final frame timestamp is 7927.52 ms. The [JSON array](outputs/demo-video-v1/predictions.json)
  passes both Python validation and the browser's `parseFrameResult` boundary.
  [Visual review](outputs/demo-video-v1/prediction-review.png),
  [export audit](outputs/demo-video-v1/predictions.info.json), and
  [media provenance](outputs/demo-video-v1/media-provenance.json) accompany it.
  No segmentation ground truth or accuracy score exists for this public clip.
  Updated predictions using the then-selected checkpoint are in
  [demo-video-small-anatomy](outputs/demo-video-small-anatomy/predictions.json),
  paired with the same prepared MP4. All 120 updated frames pass both contract
  validators and preserve the earlier export's exact presentation timestamps.
  The earlier export remains available.
- **Not implemented:** live inference API, temporal tracking/smoothing, exact
  training resume, and reviewed clinical lesson content. Camera AR is a separate path.
- **Next handoff:** use [DEMO_MEDIA.md](DEMO_MEDIA.md) for the exact source,
  preparation command, media ID and attribution. Connect this clip/result array
  to the frontend and test playback alignment. The website still uses synthetic
  assets, so the complete integrated lesson remains unfinished.

### Checks and next experiment

All **169 ML tests pass**. They cover label conversion/ignore policy, leakage,
loss/metrics, safe checkpoint provenance, original-frame geometry, invalid
contours, real variable-PTS decoding, original-grid scoring, and annotation-selected
comparison crops, plus architecture compatibility, paired augmentation,
case-balanced sampling, and cloud artifact handling. The original baseline
completed validation and test evaluation; the earlier local improvement candidates
completed validation only, and the selected one has a real clip export.

The [frozen cloud experiment set](CLOUD_EXPERIMENTS.md) is complete: longer
training, augmentation, case balancing, and model capacity were evaluated
independently. The documented Dice and resolution comparisons are also complete.
Further loss or data changes need a new documented experiment; none is running
now. Keep test data and the public demo clip
out of tuning. This test split has already been inspected; further comparisons
on it are not a fresh blind test.

On this Mac, importing both OpenCV and PyAV emits duplicate AVFoundation class
warnings from their bundled libraries. Both full clip exports completed, with
no observed crash. The pinned environment is an offline prototype; isolate
decoding in a fresh process before building a persistent live service.

At that stage, the next [offline anatomy-review ZIP](MASK_REVIEW.md) was ready: **50 images from
25 new TRAIN cases, 141 SAM proposals**, excluding every first-pilot case and
all original Seg50/held-out cases. Vertex job `5340186834892750848` completed
proposal generation. The package includes START-HERE.txt and the same editor,
with zero review decisions. No new proposals enter training before returned
reviews are validated. The updated tooling passes 211 ML tests and 12 editor
core tests. The original review and selected demo checkpoint are preserved.
