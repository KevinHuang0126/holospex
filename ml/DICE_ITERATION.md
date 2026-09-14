# Dice iteration — frozen before launch, September 13, 2026

Test whether a foreground overlap objective improves small-anatomy segmentation
at the current **672 × 384** input. Acquire additional data in parallel, but
keep this loss comparison on the existing pixel-labeled dataset so new data
does not confound the result. This plan does not increase resolution.

## Matched runs

Six fresh A100 runs use seeds **42, 43, 44**, with one balanced cross-entropy
(CE) control and one CE + foreground generalized Dice candidate for each seed.
Every run gets **40 epochs**, MobileNetV3-Large DeepLabV3, generic Torchvision
COCO/VOC pretrained weights, fresh seven-channel heads, batch size 2, AdamW,
learning rate 0.0003, frozen BatchNorm, no augmentation, uniform frame sampling,
and the same CUDA container digest and data archive as the prior cloud batch.
Balanced CE uses the existing capped inverse-square-root class weights, counted
once from each selected resized training mask. Main/auxiliary weights stay 1/0.4.
CUDA is not configured for exact determinism; paired seeds reduce dependence
on a single initialization but three seeds do not establish statistical certainty.

| Run | Seed | Training objective |
| --- | ---: | --- |
| `dice-001-ce-s42` | 42 | Balanced CE |
| `dice-002-gdl-s42` | 42 | Balanced CE + GDL |
| `dice-003-ce-s43` | 43 | Balanced CE |
| `dice-004-gdl-s43` | 43 | Balanced CE + GDL |
| `dice-005-ce-s44` | 44 | Balanced CE |
| `dice-006-gdl-s44` | 44 | Balanced CE + GDL |

## Exact Dice policy

The [Sudre et al. paper](https://arxiv.org/abs/1707.03237) motivates inverse-square
volume weighting. Our foreground-only, batch-pooled variant is explicit in
`src/holospex_ml/losses.py`; it is not an exact reproduction of that paper.
Softmax includes all seven channels, while Dice sums include present foreground
classes only. Ignore index 255 is removed from targets and predictions.

For scored target volume `V_c > 0`, use `w_c = (min_present_V / V_c)^2`.
This common rescaling preserves relative inverse-square weights before smoothing.
With intersection `I_c` and predicted probability volume `P_c`,
`GDL = 1 - (2*sum(w_c*I_c)+1e-6)/(sum(w_c*(V_c+P_c))+1e-6)`.
Absent target classes have zero Dice weight; all-class CE still penalizes their
false positives. All-background batches return differentiable Dice zero;
all-ignored batches fail. Float32 accumulation protects small-volume ratios.
Each head uses `CE + 1.0*GDL`, with auxiliary contribution 0.4.

Combo epoch loss is the mean of actual optimizer-step objectives. Log CE and
Dice separately, and retain a CE diagnostic using its target-weight denominator.
Default CE-only reporting retains its previous denominator-weighted semantics.
Do not compare the numerical total CE and CE+Dice losses as equal objectives.

## Evaluation and decisions

Use **343 train / 75 validation / 74 test frames**, from **30 / 10 / 10 cases**,
with unchanged case assignments, ignored 255, and validation quarantine
`153_32700`. The archive includes all source pairs, but test is not evaluated.
Within each run, select the first best epoch by six-class foreground macro-IoU
on resized validation labels, exactly as before. Then evaluate that checkpoint
on the original annotation grid using raw logits resized before argmax.

Report every seed and paired differences, plus three-seed means and ranges:
four-class small-anatomy pooled and case-equal IoU, six-class foreground IoU/Dice,
and per-class IoU/Dice/precision/recall. Pay particular attention to cystic-artery
recall and disagreements between pooled and case-equal scores. Do not choose
only the best seed as evidence for the objective. Validation is reused and has
only ten cases; it does not establish clinical or deployment performance.

Retain the existing demo checkpoint pending these results. New source artifacts
and launch records live in ignored `outputs/dice-handoff/`; preserve the previous
cloud batch's immutable bundle and launch manifest. Run at most three A100 jobs
concurrently to fit the project CPU quota. Spot failures are separate fresh
attempts, never spliced histories; each run has a four-hour execution timeout.

## Additional data, separate from this loss comparison

Download only new official Endoscapes TRAIN cases for the box expansion; boxes
are not pixel masks or implicit background labels. Audit the official public
CholecSeg8k masks separately, preserve its license and partial anatomy mapping,
and establish cross-dataset case overlap before future training or holdout use.
Do not inflate the test set with adjacent frames or overlap from a new release.
Record actual acquisition results and unresolved access/label issues in the
data audit documents before claiming examples are usable for segmentation.

## Recorded execution and acquired data

All six jobs reached `JOB_STATE_SUCCEEDED`, completed their epoch budgets and
original-grid validation, and were collected with artifact hash verification.
See [DICE_RESULTS.md](DICE_RESULTS.md) for the measured comparison and decision.
They were submitted with one immutable source archive:
`d5aa297a2d735fd4817c66f54ca32c4568053157a80360c6a5ac193dbca81931`.
The original Seg50 data archive remains
`831c81a7d6a8bfb9e9038ce892584f7976dc9e6451026ca60ba11b66a08f56e7`.
The [launch manifest](outputs/dice-handoff/launch-manifest.json) contains every
filled configuration and submission identity. The first three jobs finished
before the final three were submitted; there were at most three active A100s.

| Run | Vertex custom job ID |
| --- | --- |
| `dice-001-ce-s42` | `6362279649933787136` |
| `dice-002-gdl-s42` | `5294926538246979584` |
| `dice-003-ce-s43` | `5178395897888768000` |
| `dice-004-gdl-s43` | `4030540942862712832` |
| `dice-005-ce-s44` | `5422153227720196096` |
| `dice-006-gdl-s44` | `8340485786256277504` |

The separate data acquisition produced [852 usable Endoscapes box-labeled/unknown
frames from 90 new training cases](DATA_EXPANSION.md) and [8,080 CholecSeg8k
image/mask pairs](ADDITIONAL_MASK_DATA.md). The Endoscapes set has 850 images
with boxes and two images whose empty annotation lists do not establish negatives.
CholecSeg8k covers only part of the Holospex anatomy; its 17 cases have no overlap
listed in the official CAMMA crosswalk. [DATA_CATALOG.json](DATA_CATALOG.json)
records the private cloud copies and integration requirements. No acquired data
entered these six runs, and no larger fully labeled test set is claimed.
