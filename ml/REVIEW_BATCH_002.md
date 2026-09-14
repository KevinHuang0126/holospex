# Second returned review: import and training

## Completed results — September 14, 2026

Both approved training jobs succeeded, completing **53 epochs each**. Independent
audits verified the data, saved checkpoints and native validation scores. The
best new checkpoint, **batch 002 seed 42, epoch 34/53**, is promoted to
[`ml/weights/current/best.pt`](weights/current/best.pt), as explicitly requested
by the project lead. It scores **52.825114156%** six-class foreground IoU,
up **0.7842 percentage points** from the previous best run012. The 75% target
was not reached.

The controller closed at **05:11:20 UTC** with both owned jobs terminal and
all results collected. This was a separate bounded batch; the completed
September 13 four-hour search remains closed. The
[paired report and chart](outputs/reviewed-batch002-20260914/reports/paired-final/RESULTS.md)
retain the detailed comparisons.

| Seed | Training data | Selected/completed epoch | Foreground IoU | Small pooled IoU | Small equal-case IoU |
| --- | --- | --- | ---: | ---: | ---: |
| 42 | Historical pilot, run012 | 34/60 | 52.0409% | 34.6151% | 28.2247% |
| 42 | Pilot + batch 002 | 34/53 | **52.8251%** | **35.7053%** | **32.2752%** |
| 43 | Historical pilot, run014 | 24/60 | 51.6340% | 34.0068% | 30.0329% |
| 43 | Pilot + batch 002 | 39/53 | 51.9172% | 34.3737% | 31.4216% |

Both seeds improve over their historical controls. Mean foreground IoU rises
from **51.8374% to 52.3712%** (+0.5338 points); mean paired gains are +0.7285
points for pooled small anatomy and +2.7196 for equal-case small anatomy.
The new two-seed sample standard deviation is 0.6420 points. Two seeds and
repeated validation selection do not establish statistical significance.

The winner improves duct, artery and plate IoU by **4.4245, 4.2092 and 2.7144
points**, respectively. **Triangle-dissection IoU falls from 39.7821% to
32.7948% (−6.9873 points)**: recall rises, but precision falls by 15.8214 points.
Case 131 loses 16.3727 points on triangle IoU and 1.6607 points on foreground
macro IoU. In the other seed, artery and plate regress by 0.4136 and 1.6137
points versus run014. These are measured tradeoffs, not uniform improvement.

Per-case macros exclude classes absent in both truth and prediction, while
absent classes with false positives score zero. Their denominators can therefore
change between models, notably for case 137; a higher case macro does not imply
every visible structure improved. Consult per-class results alongside these
case summaries. The next annotation/error analysis should focus on triangle
overprediction and its boundaries with plate and artery.

## Reviewed data and exclusions

The anatomy-scope source review from Binh contains 141 decisions: 73 accepted,
48 edited, 19 rejected and one needs-expert. All decisions passed the strict validator against the
exact 50-image, 25-case source bundle. The import preserves the submitted JSON
verbatim and all 121 approved candidate masks. Import eligibility does not
automatically enroll a mask in training.

The lead explicitly confirmed excluding two accepted masks whose notes conflict
with acceptance: artery `endoscapes-box-930342001` and triangle
`endoscapes-box-110277751`. A separate preparation resolution records the
exclusions and reasons; the reviewer's original decisions remain unchanged.
Rejected and needs-expert candidates supply no training labels. Unknown pixels
and inter-class conflicts remain ignored (source ID 255), following the existing
partial-label policy. Same-class masks are unioned; no background is invented.

Preparation retained 119 candidate masks across 46 new images from 25 cases:
407 train / 75 validation / 74 test images. The existing 510-sample base prefix
and all source files remain unchanged. New targets retain 633,893 native
foreground pixels and ignore 16,216 conflicting pixels. Per-class retained
pixels are 114,160 artery, 320,530 duct, 71,911 plate, and 127,292 triangle.
Four images without eligible supervision are omitted. Local overlap losses
remain visible: plate `11_26275` retains only 3 native pixels, and plate
`111_25250` retains 478 of 8,492 pixels. No class receives precedence.

## Training, authorization and verification

Both runs used DeepLabV3–ResNet50 with surgical MoCo initialization at 672 × 384,
balanced CE plus 0.25 main-head Lovasz, auxiliary weight 0.4, batch size 2,
cosine scheduling, learning rate 0.0003, backbone multiplier 0.1, weight decay
0.01, no augmentation and uniform sampling. Each was fresh training from the
surgical backbone with a new optimizer. Each completed 10,812 updates
(53 × 204), compared with 10,860 (60 × 181) for the reused historical controls.
Warmup remains three epochs but spans 612 versus 543 updates; checkpoint
selection has 53 versus 60 opportunities, and balanced class weights change
with the data. This comparison measures the combined recipe, not an isolated
causal effect of the new labels.

The checkpoint selected by input-grid validation was evaluated on the fixed
75-image, ten-case validation set at original 854 × 480 annotation resolution.
The promotion criterion is pooled six-class foreground macro IoU: background
is excluded from the class average but its false positives remain counted;
source 255 is ignored. The test split was not used for inference or checkpoint
selection. Segmentation predictions remain separate from reviewed lesson answers.

Automatic approval review rejected the first source upload: it requires explicit
permission to export project source and reviewed data to the configured private
Google Cloud bucket. The lead then explicitly approved both verified bundles
and at most two A100 jobs, 53 epochs each, under a two-hour overall cap.
Both bundles uploaded successfully and their cloud MD5/size checks match the
local archives; worker bootstrap also verifies SHA-256 before training. The new
authorized window was 04:25:44–06:25:44 UTC on September 14. The controller
stopped after both runs completed at 05:11:20 UTC; no old deadline was extended.

| Seed | Vertex job ID | Final state | Epochs |
| --- | --- | --- | ---: |
| 42 | 561146350624833536 | JOB_STATE_SUCCEEDED | 53/53 |
| 43 | 4395961433330810880 | JOB_STATE_SUCCEEDED | 53/53 |

The full workspace and extracted-source suites pass 323 tests. Independent
reconstruction matches all 46 native targets and every actual loader target at
672 × 384. All 1,020 inherited image/mask files and the original 510-sample
prefix are intact. The cloud staging validator verified 1,200 bound files.
Nineteen training/evaluation source files exactly match the historical controls;
validation and test split-content fingerprints are also identical. All 36
collected completion artifacts passed byte-size and SHA-256 checks. Independent
result auditing recomputed pooled and per-case metrics from integer confusion
counts and verified best/last checkpoint metadata, selected epochs and complete
schedules. This establishes artifact and metric consistency, not clinical
validity or video accuracy.

## Local model and video handoff

The promoted [local selection record](weights/current/selection.json) identifies
`review2-20260914-001-batch002-moco-lovasz-s42`, version
`2026-09-14T04:29:29.757933Z-epoch-34`. Strict CPU loading passed. Checkpoint size
is 168,351,963 bytes and SHA-256 is
`b406ed42ab0394edba22e1dde0edc2865a6346adb61c4bea7ab0bc00d08e1911`.
Use `--checkpoint ml/weights/current/best.pt` for new inference.

Historical checkpoints remain intact. Run012's SHA-256 is
`b037e85dff5f0ca552258d733cd02364717f99b57d93e6a0b10bbcd3c2ceea0f`;
run014's is `5d1c98b979b7d03c32b6ae34d024f09de138ef616130bce6519efa367e8e5a3c`.
Existing Gupta predictions keep their actual run012 attribution.

**New Gupta export: complete and verified** at
[`ml/outputs/reviewed-batch002-20260914/gupta-current/predictions.json`](outputs/reviewed-batch002-20260914/gupta-current/predictions.json).
All 1,056 results match the decoded clip's exact frame numbers, presentation
timestamps and original 1280 × 720 dimensions. Source/model checksums and all
1,056 raw masks passed verification. The actual frontend parser and matcher
accept every result; frame 240 at 10 seconds has visible predictions with no
matching-result warning. This verifies the export, not anatomical accuracy.

Load the [exact Gupta MP4](outputs/candidate-cvs-video/gupta-2023-cvs-anterior-posterior.mp4)
first and the new JSON second; choose **ML prediction**, **Learn**, threshold
**0.5** and **Show overlays**. Previously loaded browser JSON remains unchanged
until replaced. The output is 71,053,910 bytes; SHA-256:
`d38c2788b109fe057e5734d770235fa014c1636aae39ac78b94b38300af14d8b`.
See the [identity verification](outputs/reviewed-batch002-20260914/gupta-current/identity-verification.json)
and [frontend verification](outputs/reviewed-batch002-20260914/gupta-current/frontend-verification.json).

## Artifact record

Local data, weights and reviewer notes remain outside Git under
`ml/outputs/reviewed-batch002-20260914/`. The event log records import, preparation,
packaging, execution, evaluation and promotion. Source bindings:

Source archive SHA-256:
`5a41ed8a819ae6d569f8439549f1963a035692a8d4007bb03ae36a09d5185648`.
Prepared archive SHA-256:
`935e4b68e2aed4baa1251acb11f6ed02953183f1651a5fcbebd3a975e301fe62`.
Prepared manifest SHA-256:
`e9b5ca0f146f5f0a90ab1a43c4d6c412878f26ffda21bf0a20aed580fc1a89a0`.

Source review SHA-256:
`f65147d81824c0610af5fb73150d2ffa2c9c3b8bd7f457825b5f4afe6fa56214`.
Source bundle SHA-256:
`91ad669b5a0b5225468908e25a1b344808571234ad3a36a928798af54e419586`.

- [Exact imported review](outputs/reviewed-batch002-20260914/import/review.json)
- [Import receipt](outputs/reviewed-batch002-20260914/import/receipt.json)
- [Training exclusions and source bindings](outputs/reviewed-batch002-20260914/training-resolution.json)
- [Prepared manifest](outputs/reviewed-batch002-20260914/prepared/manifest.json)
- [Preparation summary](outputs/reviewed-batch002-20260914/prepared/summary.json)
- [Event log](outputs/reviewed-batch002-20260914/events.jsonl)
- [Independent target audit](outputs/reviewed-batch002-20260914/independent-data-audit/audit.json)
- [Independent result audit](outputs/reviewed-batch002-20260914/independent-data-audit/result-audit.json)
- [Paired results and chart](outputs/reviewed-batch002-20260914/reports/paired-final/RESULTS.md)
- [Frozen comparison](outputs/reviewed-batch002-20260914/frozen-comparison.json)
- [Concrete launch plan](outputs/reviewed-batch002-20260914/launch-plan.json)
- [Local model selection](weights/current/selection.json)

[Controller state](outputs/reviewed-batch002-20260914/state.json) and
[cloud authorization](outputs/reviewed-batch002-20260914/cloud-authorization.json)
retain the exact deadline, launch budget, configuration hashes and source bindings.
