# Holospex experiment log

Consolidated on **September 13, 2026** from the saved local configurations,
histories, metrics, cloud completion records, and experiment reports.
The initial inventory covers the completed human-reviewed-data comparison,
whose record was generated at **19:20 UTC**. The subsequent four-hour autonomous
search is appended in **Iteration 8**, with its own evidence and
[live status](outputs/autonomous-20260913/STATUS.md).

## Position before the autonomous search (19:20 UTC)

- **25 completed segmentation training runs:** four local runs, five initial
  cloud runs, six loss-comparison runs, two resolution-only runs, and two runs
  adding unreviewed SAM masks, plus six reviewed-data comparison runs, totaling
  **848 completed epochs**. Reused controls
  are counted once. Every persisted history reaches its configured budget;
  no persisted partial or smoke-training history was found in this inventory.
- **Demo checkpoint:** [small-004-resolution/best.pt](outputs/small-004-resolution/best.pt),
  epoch 9 of 12, at 672 × 384. No later experiment has changed the recorded demo selection.
- **Strongest next replication candidate from the SAM pilot:**
  `sam-resolution-001-896x512-s42`. It improves the three main aggregates and
  artery recall over its same-resolution control, but is a single-seed result
  trained partly on model-generated labels. This recommendation concerns the
  SAM condition; the rejected 896 × 512 original-label checkpoint is a different
  trained model. Paired original-label/SAM replication remains outstanding.
- **Human-reviewed expansion:** six fresh A100 runs compared the original data
  with 49 approved masks across 18 additional images. All three paired seeds
  improved small pooled IoU (**24.40 → 27.23**) and equal-case IoU
  (**22.59 → 24.65**) on average. The combined manifest has
  **361 train / 75 validation / 74 test images**; optimizer-update budgets were
  matched within 0.22%. All 93 artifacts passed verification.
- **Next labeling step:** review another 40–60 images across 20–30 new TRAIN
  cases, then repeat the controlled comparison. Artery/duct overlap improved,
  while plate boundaries and duct/plate precision need attention. This is a
  bounded follow-up recommendation, not a launched review batch or demo promotion.
- **CholecSeg8k:** acquired and audited, but not used in the completed training runs.

The unreviewed SAM training trials and the later human-reviewed targets are
different experiments. The former used all 52 generated proposals in 20 images;
the latter excludes uncertain/rejected candidates and incorporates corrections.

## How to read the results

**IoU** measures overlap between predicted and annotated pixels; higher is better.
**Small pooled IoU** averages four class IoUs after combining pixels across
validation frames: cystic duct, cystic artery, cystic plate, and hepatocystic
triangle dissection. **Small equal-case IoU** first aggregates within each video
and then weights defined class scores equally across videos. It prevents a video
with more annotated pixels from dominating the result. **Foreground IoU** also
includes gallbladder and tools, excluding background.

**Recall** is the share of annotated pixels recovered. **Precision** is the share
of predicted pixels matching annotations. Higher recall can accompany more false
positives; therefore a recall gain alone does not establish a better segmenter.
All metric values below are percentages; changes are percentage points (pp).

Except for the explicitly marked initial baseline table, comparisons use the
same **75 validation frames from 10 cases at the original 854 × 480 mask grid**.
Logits are resized before argmax, with no confidence or polygon display filter.
Absent truth with false positives scores zero; absence of both is undefined and
excluded according to the saved metric policy. Each run chooses its checkpoint
using the first best six-class foreground validation IoU at its own input size.

Validation was repeatedly used for checkpoint and experiment selection. These
results are exploratory; they are not independent test estimates, reviewed lesson
answers, CVS assessments, or clinical validation. Most comparisons use one seed;
the Dice and human-reviewed-data experiments each use three. CUDA runs were not
configured for exact determinism.

## Shared starting point

- **Data:** all 493 public Endoscapes-Seg50 labeled frames were acquired. After
  quarantining validation frame `153_32700` for unexplained source ID 7, the
  usable split is **343 train / 75 validation / 74 test**, from **30 / 10 / 10
  separate surgical videos**. Training runs keep those holdouts unchanged.
- **Labels:** seven channels: background, gallbladder, duct, artery, plate,
  triangle dissection, and tool. Source PNG IDs are explicitly remapped to model
  indices. Source 255 is ignored, including in loss and metrics.
- **Model:** DeepLabV3 with MobileNetV3-Large unless specified otherwise.
  Generic Torchvision `COCO_WITH_VOC_LABELS_V1` initialization, fresh
  seven-channel main/auxiliary heads, AdamW, learning rate 0.0003, batch size 2,
  ImageNet normalization, and frozen BatchNorm. Runs start fresh; they do not
  resume an earlier anatomy checkpoint.
- **Default loss:** balanced cross-entropy (CE), except the first unweighted
  baseline and explicit generalized Dice variants. Class weights use capped
  inverse-square-root training pixel frequencies; no held-out pixels contribute.
  Masks resize by nearest neighbor, images bilinearly. Resolution changes also
  recalculate class weights with the same formula.
- **Default sampling:** uniform frames, no augmentation; seed 42 unless listed.
  Main and auxiliary head losses have weights 1.0 and 0.4.
- **Compute:** initial local runs used Apple M4 MPS. Cloud training used NVIDIA
  A100s through Vertex AI in `us-central1`, project `eastwest72hack26bos-501`.
  Cloud and Mac framework versions differ; matching settings do not guarantee
  identical optimization trajectories.

Source details: [training workflow](TRAINING.md), [run status](STATUS.md), and
[cloud execution/retrieval runbook](CLOUD_TRAINING.md).

## Iteration 1 — Establish a baseline and address class imbalance

**September 12. Question:** can the seven-class model learn useful segmentation,
and does class weighting help the small structures?

Two fresh local MPS runs used 448 × 256 input and three epochs. About **97.01%**
of scored training pixels belonged to background, gallbladder, or tools, motivating
the second run's class weighting. Both selected epoch 3.

This table uses **448 × 256 evaluation**, not original-grid evaluation:

| Run | Loss | Epochs / selected | Validation foreground IoU | Validation Dice | Test foreground IoU / Dice |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline-001` | Unweighted CE | 3 / 3 | 31.47 | 39.27 | Not evaluated |
| `baseline-002-balanced` | Balanced CE | 3 / 3 | 36.76 | 48.13 | 35.57 / 46.01 |

**Finding:** weighting improved validation foreground IoU by **5.29 pp**.
The balanced model became the control for later work. It is the only model with
a recorded test-set evaluation in this log. The test split has therefore already
been inspected and should not be described as a fresh blind holdout.

Visual diagnostics still found false-positive plate regions and overextended
gallbladder predictions. The model's softmax scores are uncalibrated.

Evidence: [baseline configuration](outputs/baseline-001/config.json),
[baseline validation](outputs/baseline-001/metrics-validation.json),
[balanced configuration](outputs/baseline-002-balanced/config.json),
[balanced validation](outputs/baseline-002-balanced/metrics-validation.json), and
[balanced test report](outputs/baseline-002-balanced/metrics-test.json).

## Iteration 2 — More training, then more input pixels on the Mac

**September 12. Question:** do longer training and less downsampling improve
the four small structures? Two additional runs isolated budget and input size.

| Run | Input | Epochs / selected | Small pooled IoU | Small equal-case IoU | Foreground IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline-002-balanced` — reused control | 448 × 256 | 3 / 3 | 18.60 | 16.20 | 36.79 |
| `small-003-longer` | 448 × 256 | 12 / 9 | 22.43 | 19.73 | 40.64 |
| `small-004-resolution` | 672 × 384 | 12 / 9 | 24.36 | 25.73 | 42.94 |

**Finding:** longer training helped; moving to 672 × 384 further improved all
three aggregate scores. All four small-class pooled/equal-case IoUs improved
against the three-epoch control. Artery precision rose **13.61 → 23.21**, while
recall fell **42.37 → 34.27**. Individual images still regressed.

**Decision:** select `small-004-resolution/best.pt` for the demo. Its SHA-256 is
`448f77c258d054107f4cdde75a051b92c373503e7b0470d01bb4c27d27ede838`.

The mask-resizing audit found no entire focus-class annotation lost at 448 × 256;
some tiny disconnected components disappeared, all smaller than 16 original
pixels. Median training artery area grew from 896 to 2,017 input pixels at
672 × 384. This supported testing detail preservation without claiming that
whole annotations had disappeared. An 896 × 512 grid was audited here and only
trained later in iteration 5.

Evidence: [full local comparison](SMALL_ANATOMY.md),
[longer-run metrics](outputs/small-003-longer/metrics-val-original.json),
[selected-run metrics](outputs/small-004-resolution/metrics-val-original.json),
[selection record](outputs/small-anatomy-selection.json), and
[visual comparison](outputs/small-004-resolution/comparison-small.png).

## Iteration 3 — Move to A100 and test budget, augmentation, sampling, backbone

**September 13.** Five fresh runs used 672 × 384 input and balanced CE. The
first reproduces the 12-epoch local recipe on CUDA; the longer control then
supports three independent 40-epoch comparisons.

| Run | Change | Epochs / selected | Small pooled IoU | Small equal-case IoU | Foreground IoU | Artery recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cloud-001-control` | CUDA migration control | 12 / 7 | 23.53 | 21.89 | 42.01 | 24.61 |
| `cloud-002-longer` | 40-epoch budget | 40 / 38 | 26.36 | 25.35 | 44.77 | 24.63 |
| `cloud-003-augmentation` | Mild paired augmentation | 40 / 32 | 24.79 | 20.72 | 43.83 | 20.78 |
| `cloud-004-case-balanced` | Equal expected case sampling | 40 / 38 | 23.95 | 22.04 | 43.19 | 19.33 |
| `cloud-005-resnet50` | DeepLabV3–ResNet50 | 40 / 32 | 25.46 | 23.38 | 44.26 | 29.68 |

The augmentation recipe was a paired horizontal flip with probability 0.5,
plus image-only brightness/contrast factors in [0.9, 1.1]. Case balancing sampled
343 frames with replacement per epoch, using inverse frames-per-video weights.
The larger backbone used its corresponding generic pretrained initialization.

**Findings and decisions:**

- **Longer training:** strongest initial cloud aggregate result. Against the
  12-epoch CUDA control, small pooled IoU gained 2.83 pp and equal-case IoU
  gained 3.47 pp. It remained a research candidate because case-equal IoU and
  artery recall were below the selected Mac checkpoint.
- **Augmentation:** plate and triangle IoU improved versus the longer control,
  while duct and artery performance declined. Do not promote this recipe.
- **Case balancing:** plate recall rose to 37.23%, but artery IoU fell to 10.26%.
  Do not promote this sampling configuration.
- **ResNet50:** artery IoU rose to 17.62%, but plate/duct tradeoffs and lower
  aggregate scores did not justify promotion. Local inference was much slower.

All five jobs succeeded and their artifacts were collected with hash checks.
The complete data and ordered validation identities match across runs. Early
CUDA histories differed even with the same seed, so budget-only comparisons
also contain run-to-run variation. No test evaluation was added.

Evidence: [frozen plan](CLOUD_EXPERIMENTS.md), [full results](CLOUD_RESULTS.md), and
[launch/configuration inventory](outputs/cloud-handoff/launch-manifest.json).

## Iteration 4 — Balanced CE versus CE + generalized Dice, three seeds

**September 13. Question:** can an overlap-based loss help class imbalance before
increasing resolution again? Six fresh 40-epoch runs held input at 672 × 384,
using seeds 42, 43, and 44, with one CE and one CE + GDL run for each seed.

The tested GDL is foreground-only, batch-pooled, with inverse-square target-volume
weights, zero weight for absent target classes, ignored pixels excluded, float32
accumulation, and smoothing 1e-6. Each head uses **CE + 1.0 × GDL**; auxiliary
weight remains 0.4. CE continues to penalize absent-class false positives.
This records one specific Dice variant and coefficient, not every possible Dice loss.

| Run | Loss / seed | Selected epoch / 40 | Small pooled IoU | Small equal-case IoU | Foreground IoU | Artery recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dice-001-ce-s42` | CE / 42 | 16 | 25.49 | 22.57 | 44.06 | 20.86 |
| `dice-002-gdl-s42` | CE + GDL / 42 | 20 | 25.12 | 21.52 | 43.49 | 22.48 |
| `dice-003-ce-s43` | CE / 43 | 31 | 23.45 | 21.91 | 43.13 | 23.34 |
| `dice-004-gdl-s43` | CE + GDL / 43 | 23 | 24.71 | 21.05 | 42.87 | 21.56 |
| `dice-005-ce-s44` | CE / 44 | 29 | 26.44 | 22.79 | 45.06 | 18.99 |
| `dice-006-gdl-s44` | CE + GDL / 44 | 30 | 23.95 | 21.89 | 43.08 | 24.59 |
| Mean, CE | Three seeds | — | 25.13 | 22.42 | 44.08 | 21.06 |
| Mean, CE + GDL | Three seeds | — | 24.60 | 21.49 | 43.15 | 22.88 |

**Finding:** equal-case small IoU and foreground IoU declined in all three
matched pairs. Mean artery IoU improved **13.95 → 15.86**, but mean plate IoU
fell **15.39 → 11.14** and plate recall **24.12 → 15.63**. The three-seed ranges
are variability observations, not confidence intervals.

**Decision:** retain balanced CE as the control. Do not promote the tested GDL
configuration. All six jobs completed with 13 verified artifacts per run;
independent confusion-matrix recomputation found no metric discrepancies.
New datasets were acquired separately and did not enter these runs.

Evidence: [exact formula and frozen settings](DICE_ITERATION.md),
[full results](DICE_RESULTS.md), [comparison JSON](outputs/dice-handoff/comparison.json),
[independent audit](outputs/dice-handoff/final-independent-audit.json), and
[visual examples](outputs/dice-handoff/seed42-comparison-small.png).

## Iteration 5 — Higher resolution with the original labels

**September 13.** Two fresh 40-epoch A100 runs used 896 × 512 and 1120 × 640,
holding the remaining recipe to the 40-epoch balanced-CE cloud control. Both
used the original 343 training frames; no SAM or human-reviewed labels were added.

| Run | Input | Selected epoch / 40 | Small pooled IoU | Small equal-case IoU | Foreground IoU | Artery recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cloud-002-longer` — reused control | 672 × 384 | 38 | 26.36 | 25.35 | 44.77 | 24.63 |
| `resolution-001-896x512` | 896 × 512 | 35 | 25.46 | 20.82 | 44.80 | 20.02 |
| `resolution-002-1120x640` | 1120 × 640 | 34 | 27.20 | 24.76 | 45.82 | 43.87 |

**Findings:** 896 × 512 lowered small-anatomy metrics and artery recall. At
1120 × 640, small pooled IoU gained 0.84 pp and foreground IoU 1.05 pp, but
equal-case IoU fell 0.59 pp. Artery recall rose sharply while precision fell
**30.39 → 17.59**; artery IoU fell **15.75 → 14.36** and plate IoU
**18.28 → 14.03**. Triangle IoU improved **27.96 → 36.98**; duct IoU was flat.

**Decision:** reject the 896 × 512 original-label candidate; retain 1120 × 640
for research, with no demo replacement. This decision is specific to these
original-label checkpoints; the SAM condition is recorded in iteration 6.
Both jobs completed with 13 verified artifacts per run. The 1120 × 640 history
includes transient loss spikes;
the selected epoch remains 34, rather than the final epoch.

These grids contain 1.78× and 2.78× the pixels of 672 × 384. Source images are
854 × 480, so the larger grids also interpolate beyond native dimensions; they
do not acquire new optical detail. All tested training grids use a 7:4 aspect
ratio. This was not an exact-native 854 × 480 training trial.

Evidence: [resolution report](RESOLUTION_ITERATION.md),
[comparison JSON](outputs/resolution-handoff/comparison.json),
[launch record](outputs/resolution-handoff/launch-manifest.json), and
[annotation-selected visual sheet](outputs/resolution-002-1120x640/comparison-small.png).

## Iteration 6 — Add unreviewed SAM proposals at both larger sizes

**September 13.** A separate experiment added **20 images / 20 new TRAIN cases**
with **52 unreviewed SAM 2.1 proposals** to the original training set:
**363 train / 75 validation / 74 test**. It reused the same two resolutions and
40-epoch, seed-42 balanced-CE recipe. These were additional DeepLabV3 training
runs; SAM itself was used for proposal inference and was not fine-tuned here.

Same-class proposals were unioned. Pixels outside all positives and cross-class
conflicts were set to ignore 255. Proposed regions were not clipped to boxes,
and uncovered regions did not become background. Training grew from **343 images
in 30 cases to 363 images in 50 cases**. At batch size 2, each epoch increased
from **172 to 182 optimizer updates**: **6,880 → 7,280 updates** over 40 epochs,
an additional 400 updates (5.81%). Steps were not matched across data conditions.

The balanced-CE formula stayed fixed, but its class weights were recalculated
from the enlarged training set. The seeded shuffle also changed with dataset
size; seed 42 does not preserve the original batch order. This experiment
measures the combined effect of added cases, partial SAM supervision, derived
weights, sample order, and additional updates. It cannot attribute the gains to
SAM boundary quality alone.

Combined view of the five 40-epoch cloud models, including reused controls:

| Run | Input | Selected epoch / 40 | Small pooled IoU | Small equal-case IoU | Foreground IoU | Artery recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cloud-002-longer` — original labels | 672 × 384 | 38 | 26.36 | 25.35 | 44.77 | 24.63 |
| `resolution-001-896x512` — original labels | 896 × 512 | 35 | 25.46 | 20.82 | 44.80 | 20.02 |
| `sam-resolution-001-896x512-s42` | 896 × 512 | 13 | 27.57 | 25.23 | 46.00 | 42.38 |
| `resolution-002-1120x640` — original labels | 1120 × 640 | 34 | 27.20 | 24.76 | 45.82 | 43.87 |
| `sam-resolution-002-1120x640-s42` | 1120 × 640 | 23 | 28.10 | 25.28 | 45.60 | 24.10 |

Against the **same-resolution original-label controls**:

| Input | Change in small pooled IoU | Change in equal-case IoU | Change in foreground IoU | Change in artery recall |
| --- | ---: | ---: | ---: | ---: |
| 896 × 512 | +2.11 | +4.42 | +1.21 | +22.37 |
| 1120 × 640 | +0.90 | +0.52 | -0.21 | -19.77 |

**Finding:** at 896 × 512, artery IoU rose **13.48 → 16.74**, plate IoU
**11.76 → 18.07**, and duct IoU **41.97 → 42.61**; triangle/tool IoU declined.
Artery recall rose **20.02 → 42.38**, while precision fell **29.21 → 21.68**.
At 1120 × 640, small-class overlap improved overall, but gallbladder/tool IoU
and artery recall declined: artery precision rose **17.59 → 32.89**, while
recall fell **43.87 → 24.10**. Compared directly with the 896 SAM run, 1120 adds
only 0.53 pp small pooled IoU and 0.05 pp equal-case IoU, while foreground IoU
falls 0.40 pp and artery recall falls from 42.38% to 24.10%.

**Decision:** the 896 × 512 SAM trial is the stronger replication candidate.
The 1120 SAM trial has the highest recorded small pooled IoU (28.10%), but that
single aggregate does not make it the demo winner. Both remain experimental;
neither incorporates the teammate's subsequent corrections. Both jobs succeeded,
with 14 verified artifacts per run and a separate independent results audit.

**Why the recommendations differ:** the resolution-only and SAM reports contain
identical original-label control scores and checkpoint hashes. All five models
use the same 75 validation frames, 10 cases, original 854 × 480 scoring grid,
and ignore-255 policy. The differing recommendations concern different training
conditions, rather than a discrepancy in the measurements. Rejecting the
original-label 896 checkpoint does not establish that the resolution is unsuitable
after adding SAM supervision.

The best selected epochs also changed **35 → 13 at 896** and **34 → 23 at 1120**;
all runs still completed their full 40-epoch budgets. Each SAM run improved
per-video small-class macro IoU in **6 of 10 validation cases** and regressed
in **4**, so the gains were not uniform. Among these five 40-epoch cloud models,
the 672 control still
has the highest small equal-case IoU (**25.35**, versus **25.23 / 25.28** for SAM).
The separate 12-epoch local demo checkpoint retains **25.73** on that metric.

**Next controlled check:** repeat paired original-label versus unreviewed-SAM
conditions at additional matched seeds, prioritizing 896 × 512 for replication.
Keep the label condition explicit: the later human-reviewed targets are tested
separately in iteration 7. No reliable superiority or demo promotion is established
by the current single-seed SAM trials.

Evidence: [frozen SAM plan](outputs/sam-resolution-handoff/EXPERIMENT.md),
[full SAM results](outputs/sam-resolution-handoff/RESULTS.md),
[comparison JSON](outputs/sam-resolution-handoff/comparison.json),
[independent audit](outputs/sam-resolution-handoff/independent-results-audit.json),
and [job/collection record](outputs/sam-resolution-handoff/handoff-status.json).

## Iteration 7 — Human-reviewed partial labels, three paired seeds

**September 13. Question:** do the teammate's reviewed labels improve small
anatomy enough to justify labeling more, compared with spending nearly the same
training budget on the original data?

Six fresh A100 runs paired seeds **42, 43, and 44**. Controls used the original
**343 train images / 30 cases**; candidates added **49 accepted/corrected masks
across 18 new TRAIN images/cases**, yielding **361 images / 48 cases**. These
are the reviewed targets, not the 52 unreviewed proposals used in iteration 6.
Only uncontested reviewed foreground was labeled; unknown and conflicting
pixels remained ignored. All original image/mask bytes and validation/test
identities were preserved.

All runs used DeepLabV3–MobileNetV3-Large at **672 × 384**, balanced CE, fresh
generic pretrained initialization, batch size 2, learning rate 0.0003, frozen
BatchNorm, uniform sampling, and no augmentation. Controls completed **42 epochs
/ 7,224 updates**; reviewed-data runs completed **40 epochs / 7,240 updates**.
This **0.22% update-budget difference** was frozen before training. Controls
had two more validation checkpoint-selection opportunities. Class weights were
recalculated with the same training-only formula; sample order also changes
with dataset size. Fresh controls avoid relying on an older CUDA realization.

| Run | Data / seed | Epochs / selected | Small pooled IoU | Small equal-case IoU | Foreground IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| `reviewed-001-base-s42` | Original / 42 | 42 / 34 | 23.78 | 22.72 | 42.62 |
| `reviewed-002-added-s42` | Reviewed addition / 42 | 40 / 33 | 27.65 | 26.43 | 45.87 |
| `reviewed-003-base-s43` | Original / 43 | 42 / 26 | 24.32 | 23.00 | 43.35 |
| `reviewed-004-added-s43` | Reviewed addition / 43 | 40 / 38 | 26.50 | 23.62 | 44.79 |
| `reviewed-005-base-s44` | Original / 44 | 42 / 23 | 25.11 | 22.05 | 43.80 |
| `reviewed-006-added-s44` | Reviewed addition / 44 | 40 / 38 | 27.53 | 23.91 | 45.20 |
| Mean, original | Three seeds | — | 24.40 | 22.59 | 43.26 |
| Mean, reviewed addition | Three seeds | — | 27.23 | 24.65 | 45.29 |

**Finding:** all three paired seeds improved both small-anatomy aggregates.
Mean gains were **+2.83 pp pooled**, **+2.06 pp equal-case**, and **+2.03 pp
foreground IoU**. Pooled gains ranged from +2.18 to +3.88 pp; equal-case gains
ranged from +0.62 to +3.71 pp. This met the predeclared practical criterion:
at least +1 pp mean improvement in both small metrics, with positive changes
in both metrics in at least two pairs. It is not a significance test.

| Small anatomy | Mean pooled IoU: original → reviewed | Mean precision: original → reviewed | Mean recall: original → reviewed |
| --- | ---: | ---: | ---: |
| Cystic duct | 37.95 → 42.11 | 66.60 → 62.43 | 46.88 → 56.69 |
| Cystic artery | 12.73 → 17.67 | 28.32 → 31.03 | 18.72 → 29.59 |
| Cystic plate | 18.94 → 19.28 | 37.24 → 32.59 | 27.89 → 32.44 |
| Triangle dissection | 27.98 → 29.85 | 37.99 → 38.98 | 51.72 → 57.32 |

Artery was the clearest gain: **+4.93 pp IoU and +10.87 pp recall**. Duct IoU
also improved, but precision declined in every pair, averaging **-4.18 pp**.
Plate pooled IoU declined in two pairs and improved only +0.33 pp on average.
Its equal-case IoU fell **18.29 → 17.19**, and equal-case precision fell
**36.56 → 28.55**, with precision declines in all three pairs. Triangle's mean
improved, but seed 42 traded -6.40 pp precision for +20.12 pp recall.

**Decision:** another bounded batch of **40–60 reviewed images across 20–30
new training cases** is warranted, with particular attention to plate boundaries
and duct/plate false positives, followed by the same controlled evaluation.
Keep the selected demo checkpoint unchanged and retain the new checkpoints as
candidates. This is pilot evidence from ten repeatedly used validation cases;
it does not isolate the effect of human correction from additional images and
supervision, because no matched unreviewed-proposal arm was run here.

The prepared cloud adapter verified the exact source package before rebasing
operational paths. Both arms recorded actual image/mask content fingerprints.
All six jobs succeeded; **93 artifacts, 12 selected/last checkpoints, and 246
epoch records** passed the independent audit. Original-grid pooled, per-case,
and equal-case metrics were recomputed from confusion matrices. No test
evaluation or public-demo scoring was added.

The seed-42 annotation-selected visual check retained mixed examples: whole-frame
focus IoUs were duct **0.587 → 0.477**, artery **0.044 → 0.312**, plate
**0.000 → 0.000**, and triangle **0.061 → 0.192**. Crops are display-only;
these four examples do not replace the aggregate result.

Evidence: [frozen plan](REVIEWED_DATA_ITERATION.md),
[full results and checkpoint hashes](REVIEWED_DATA_RESULTS.md),
[comparison JSON](outputs/reviewed-handoff/comparison.json),
[launch/configuration inventory](outputs/reviewed-handoff/launch-manifest.json),
[independent audit](outputs/reviewed-handoff/final-independent-audit.json),
[final job states](outputs/reviewed-handoff/final-job-status.json), and
[visual examples](outputs/reviewed-handoff/seed42-comparison-small.png).

## Data and annotation work completed alongside training

### Endoscapes expansion

Acquired 869 additional original TRAIN frames from 90 cases. Seventeen repeated
black frames were quarantined, leaving **852 usable images and 3,951 boxes**.
Of those images, 850 have boxes; two have unknown annotation completeness.
The added boxes include 696 duct, 434 artery, 288 plate, 311 triangle,
839 gallbladder, and 1,383 tool annotations. Boxes are not pixel segmentation labels.

Audits verified source membership, dimensions, geometry, CRC/SHA checks, absence
of held-out case overlap, and zero usable image-hash matches with existing Seg50.
A pre-existing group of 22 identical blank Seg50 images (5 train / 2 validation /
15 test) was recorded and left fixed across comparisons. Public Seg201 pixel
masks remain unavailable; access was not bypassed. The download resumed after
an interrupted acquisition and the final inventory was verified.

Evidence: [expansion report](DATA_EXPANSION.md),
[source-box visual audit](outputs/bbox-expansion-audit/source-box-audit.png), and
[private-cloud data catalog](DATA_CATALOG.json).

### CholecSeg8k acquisition and mapping

Acquired official version 11: **8,080 images with semantic masks**, from 17 videos
and 101 clips. All 32,320 original image/mask-related files were CRC/hash audited.
The conservative mapping supplies gallbladder, cystic duct, and tools; it supplies
no cystic artery, plate, or triangle-dissection labels. **7,331 frames** contain
at least one mapped target. Only **248 frames** contain duct labels, with 244
from one video and two frames from another containing just three duct pixels total.

The official CAMMA crosswalk lists no Endoscapes case overlap for the acquired
17 videos. This does not create an independent test set or eliminate every
possible duplicate. Values outside confirmed target IDs would need to be ignored
under the proposed partial-label policy. Original data and provenance were copied
to private GCP storage. **No CholecSeg8k model training has been run.**

ATLAS-120k was identified as a larger possible follow-up with partial class
coverage; access is gated and it was not downloaded. Dataset access, source
licenses, ontology limits, and visual audits are in [ADDITIONAL_MASK_DATA.md](ADDITIONAL_MASK_DATA.md).

### SAM proposal generation and review interface

A generic **SAM 2.1 large** model used official source boxes on 20 new TRAIN
images to generate **52 proposals: 16 arteries, 16 ducts, 9 plates, 11 triangles**.
Proposal generation took 9.04 seconds on an A100, excluding startup/transfer.
Vertex job `4112639277085491200` succeeded; 83 artifacts were verified.
Thirty-six proposals extended outside their boxes and were flagged for review.
The source box supplies the anatomy class; SAM does not verify anatomical identity.

Built and exercised the portable HTML reviewer: overlay/box toggles, zoom,
brush/erase/undo, explicit named decisions, notes for rejection/uncertainty,
JSON export/resume, and strict Python validation/import. Technical-scope checks
used fixtures and did not create anatomy-approved labels. The delivered package
works as an offline copy; it is not a shared hosted labeling service.

Evidence: [review handoff](MASK_REVIEW.md),
[portable review ZIP](outputs/holospex-mask-review-pilot-001.zip), and
[proposal generation audit](outputs/review-pilot-cloud/generation-audit.json).

### Returned human review and partial-target preparation

The first returned anatomy review contains all **52 decisions**:
**36 accepted unchanged, 13 edited and approved, 2 needs-expert, 1 rejected**.
Imported **49 approved masks across 18 images/cases**: 14 artery, 15 duct,
9 plate, and 11 triangle masks. Seven of the eleven triangle proposals were
expanded by the reviewer. Across all edits, 52,584 pixels were added and 20,823
removed. These are annotation corrections, not measured model improvements.

The lead confirmed that two flagged notes described errors already fixed and
selected ignoring cross-class overlap. Same-class masks are unioned. Exactly
5,457 conflicting pixels are ignored; uncovered pixels remain unknown/ignored.
The resulting targets retain **323,094 reviewed foreground pixels** and add 18
images to the original 343, producing **361 / 75 / 74** samples. Imported binary
candidate masks use `255 = positive`; converted training PNGs instead use explicit
anatomy source IDs and `255 = ignore`. They are not interchangeable file formats.

Native hashes, actual loader targets at four resolutions, unchanged original
samples/holdouts, and zero ignored-pixel CE/Dice gradients were verified. The
default Vertex entry point regenerates Seg50 and would omit these additions.
The later verified prepared-package adapter explicitly enrolled the combined
manifest for the six completed runs in iteration 7. Original review and
preparation snapshots retain their historical preparation-time status.

Evidence: [review results](REVIEW_PILOT_RESULTS.md), [target policy](PARTIAL_TRAINING.md),
[prepared manifest](outputs/reviewed-partial-001/manifest.json), and
[preparation summary](outputs/reviewed-partial-001/summary.json).

## Inference, rendering, and runtime trials

- Prepared an eight-second, 1280 × 720 public surgical-video excerpt and exported
  **120 consecutive direct frame predictions** with the baseline and selected
  small-anatomy checkpoints. Actual decoded presentation timestamps were retained;
  the final frame is at 7927.52 ms. Exports passed the Python and browser contract
  boundaries. The clip has no segmentation ground truth, so these are integration
  trials with no accuracy score. See [media provenance/handoff](DEMO_MEDIA.md).
- Built a [three-case labeled-data viewer](outputs/labeled-samples-v1/README.md)
  with original images, exact rasters, approximate polygons, opacity controls,
  class mappings, and provenance. These are supplied dataset annotations, not
  model outputs or newly reviewed lesson answers.
- Generated annotation-selected comparison sheets for local, cloud, Dice,
  resolution, and reviewed-data checkpoints. They preserve examples that regress. Crops are for
  display; class metrics cover the full original frame.

The two timing protocols below differ and must not be merged into one ranking:

| Benchmark | Device/protocol | Model/input | Median / p90 adapter latency |
| --- | --- | --- | ---: |
| Initial cloud-checkpoint handoff | M4 MPS; 3 warmups, 10 calls | `cloud-002-longer`, MobileNetV3, 672 × 384 | 53.39 / 54.25 ms |
| Initial cloud-checkpoint handoff | M4 MPS; 3 warmups, 10 calls | `cloud-005-resnet50`, ResNet50, 672 × 384 | 486.03 / 738.38 ms |
| Resolution handoff | Local CPU; 5 warmups, 30 calls | `cloud-002-longer`, 672 × 384 | 121.89 / 123.23 ms |
| Resolution handoff | Local CPU; 5 warmups, 30 calls | `resolution-001-896x512`, 896 × 512 | 165.28 / 168.07 ms |
| Resolution handoff | Local CPU; 5 warmups, 30 calls | `resolution-002-1120x640`, 1120 × 640 | 227.07 / 231.51 ms |

Each uses one fixed 854 × 480 image and includes decode, preprocessing, inference,
original-grid geometry extraction, and schema validation. Model loading,
network/camera capture, browser rendering, and display are excluded. They do not
measure phone, glasses, A100 inference, or streaming frame rate. The resolution
benchmark initially requested MPS but the process could not access it; the
reported completed comparison therefore used CPU. Early shorter CPU samples
were superseded by the recorded 30-call benchmark.
No CPU latency measurement for either SAM checkpoint is recorded here; the
resolution-handoff timings above belong to the original-label checkpoints.

Mean A100 epoch times for the original-label resolution comparison were
**16.79 / 20.28 / 25.67 seconds** at 672 / 896 / 1120 widths. These include epoch
training and validation, excluding provisioning and final evaluation/transfer.
The SAM trials' total epoch times were **14.43 / 17.88 minutes** for 40 epochs;
they also perform more updates per epoch because of the extra images.

Evidence: [MPS benchmark](outputs/cloud-handoff/inference-benchmark.json) and
[CPU benchmark](outputs/resolution-handoff/inference-benchmark.json).

## Execution evidence and limitations

Cloud runs preserved immutable bundles, unique attempt IDs, selected/last
checkpoints, histories, configs, original-grid metrics, and checksum inventories.
Exact optimizer/RNG/sampler resume is not implemented. A restart is a separate
attempt; histories must not be spliced together.

The first source-packaging attempt for the resolution sweep omitted review helper
modules while including their tests. It failed before upload/submission; the
fixed extracted bundle passed 169 tests and was used for both completed jobs.
This packaging failure is not an additional training trial.

Recorded validation checkpoints belong to different source snapshots:

| Snapshot | Recorded checks |
| --- | --- |
| Initial five-cloud-run implementation | 118 ML tests |
| Dice source bundle / final Dice workspace | 135 / 143 ML tests |
| Original review UI implementation | 168 ML tests + 12 review-core tests and browser exercise |
| Resolution source bundle / final resolution workspace | 169 ML tests |
| Isolated SAM-resolution source | 184 ML tests plus data preflight and independent results audit |
| Human-reviewed partial-target preparation | 181 workspace ML tests, including 12 target-converter/loader tests |
| Reviewed-data cloud source / final analysis workspace | 192 extracted-source / 205 workspace ML tests; independent 93-artifact results audit |

These are historical source-specific counts, not a monotonic coverage score.
Writing this consolidated document does not rerun training or the model tests.
Published source reports may contain earlier next-step language; this log records
the later SAM experiments, returned review, and reviewed-data training explicitly. Generated datasets,
weights, review payloads, and run artifacts remain Git-ignored; links under
`outputs/` refer to this local workspace and may be absent in another clone.

## Cloud job register

All jobs below have saved successful completion evidence. This is a historical
register, not a live project-wide query for unrelated jobs.

| Run | Vertex custom job ID |
| --- | --- |
| `cloud-001-control` | `3017512501681061888` |
| `cloud-002-longer` | `6848949884666511360` |
| `cloud-003-augmentation` | `2909989060577591296` |
| `cloud-004-case-balanced` | `6893422930986795008` |
| `cloud-005-resnet50` | `7671982716568469504` |
| `dice-001-ce-s42` | `6362279649933787136` |
| `dice-002-gdl-s42` | `5294926538246979584` |
| `dice-003-ce-s43` | `5178395897888768000` |
| `dice-004-gdl-s43` | `4030540942862712832` |
| `dice-005-ce-s44` | `5422153227720196096` |
| `dice-006-gdl-s44` | `8340485786256277504` |
| SAM proposal generation — inference only | `4112639277085491200` |
| `resolution-001-896x512` | `6125326098055036928` |
| `resolution-002-1120x640` | `3324087129830588416` |
| `sam-resolution-001-896x512-s42` | `6801991942067453952` |
| `sam-resolution-002-1120x640-s42` | `5318055864848875520` |
| `reviewed-001-base-s42` | `1003959265548828672` |
| `reviewed-002-added-s42` | `6619948000879837184` |
| `reviewed-003-base-s43` | `8155675473813176320` |
| `reviewed-004-added-s43` | `3080963118696759296` |
| `reviewed-005-base-s44` | `661685693868670976` |
| `reviewed-006-added-s44` | `8665426656636174336` |

Original local run outputs are under `outputs/RUN_NAME/`; the initial cloud,
Dice, and original-label resolution runs use `outputs/RUN_NAME/train/` for
configs, metrics, and checkpoints. SAM training results instead live under
`outputs/sam-resolution-handoff/results/RUN_NAME/train/`; reviewed-data runs use
`outputs/reviewed-handoff/results/RUN_NAME/train/`. Their exact checkpoint
hashes, source/data archive identities, and GCS URIs are preserved in the linked
comparison/launch/completion records rather than inferred from filenames.

## Still untested or incomplete

- Reviewing the next **40–60 images across 20–30 new TRAIN cases** and measuring
  whether the pilot gains persist as reviewed supervision grows. The first
  18-image reviewed-data comparison is complete; this follow-up is not launched.
- Repeating paired original-label versus unreviewed-SAM conditions at additional
  matched seeds, including the 896 × 512 replication candidate.
- Training on CholecSeg8k with partial labels; acquisition and mapping alone do
  not establish a model benefit.
- Other Dice formulations/coefficients, exact-native 854 × 480 input, combined
  augmentation/backbone/sampling sweeps, and systematic throughput tuning.
- Live camera/phone inference, temporal tracking/smoothing, and AR-glasses deployment.
- A larger independent, fully labeled evaluation set. The new TRAIN cases and
  public unlabeled clip do not expand the held-out segmentation benchmark.

When adding an iteration, record its question, control, exact data/label status,
settings changed, completed and selected epochs, original-grid/per-case metrics,
per-class tradeoffs, job/checkpoint identity, evidence links, and selection decision.
Mark planned, interrupted, and preparation-only work explicitly.

## Iteration 8 — Autonomous optimization and detail decoder search

**September 13, started 19:44:23 UTC; deadline 23:44:23 UTC.** The user authorized
autonomous training until four hours elapse or verified original-grid validation
foreground mean IoU reaches 75%. This is an exploratory search on the existing
75 validation frames, with no new test-set evaluation.

The initial candidates test low-rate warm-start refinement of the reviewed
seed-42 checkpoint, cosine scheduling at 896 × 512, a MobileNet DeepLabV3+ style
stride-four detail decoder, and a lower initial learning rate. All use the
verified 361-image reviewed-data training manifest and keep the held-out case
identities and ignored-label policy unchanged. Subsequent choices use verified
validation results. Full settings and rationale are in
[AUTONOMOUS_TRAINING.md](AUTONOMOUS_TRAINING.md).

**Status at setup:** implementation and validation in progress; no new training
result is claimed by this entry. The controller's current state and per-run
results are written to [live status](outputs/autonomous-20260913/STATUS.md),
[state and configurations](outputs/autonomous-20260913/state.json), and the
append-only [event log](outputs/autonomous-20260913/events.jsonl). Completed
results and the final selection decision will be appended after verification.

**First result, verified 20:09 UTC:** `auto-20260913-001-warm-cosine` completed
40 epochs of weight-only fine-tuning from `reviewed-002-added-s42`, using
672 × 384, balanced CE, learning rate 3e-5 with cosine decay, and unchanged
reviewed training data. The selected checkpoint was **epoch 1**. Original-grid
foreground IoU was **45.8758%**, versus **45.8717%** for its initialization
(+0.0042 pp). Small pooled IoU declined **27.6545 → 27.5146%**, and small
equal-case IoU declined **26.4280 → 25.8629%**. This is no practical improvement;
the later epochs overfit and the demo selection is unchanged.

Vertex job: `2667863406001782784`. Selected checkpoint SHA-256:
`07f9e5c89833bed895b74386e4edcea1f2a64ac0b5a574e9dfd9d031b30eaab8`.
Evidence: [native metrics](outputs/autonomous-20260913/results/auto-20260913-001-warm-cosine/train/metrics-val-original.json),
[independent audit](outputs/autonomous-20260913/results/auto-20260913-001-warm-cosine/independent-audit.json),
and [training configuration](outputs/autonomous-20260913/results/auto-20260913-001-warm-cosine/train/config.json).
The 896 × 512 scheduled run remains active, and the detail-decoder run was
submitted next as job `586918903179902976`.

**Second result, verified 20:25 UTC:** `auto-20260913-002-reviewed-896-cosine`
completed 80 epochs from generic pretrained MobileNet weights at 896 × 512,
balanced CE, and learning rate 3e-4 with cosine decay. Epoch **43** was selected.
Original-grid foreground IoU was **45.6864%**, small pooled IoU **27.1182%**,
and small equal-case IoU **26.3436%**. The larger scheduled input does not beat
the reviewed 672 × 384 baseline on any of these three aggregate metrics.
This comparison changes both resolution and schedule relative to that
historical baseline, so it does not isolate either change by itself.

Vertex job: `181876411693268992`. Selected checkpoint SHA-256:
`f3de04f96e05b3038d725a6a5320a8898a8344b8a1c42378ae4ae4adee424035`.
Evidence: [native metrics](outputs/autonomous-20260913/results/auto-20260913-002-reviewed-896-cosine/train/metrics-val-original.json),
[independent audit](outputs/autonomous-20260913/results/auto-20260913-002-reviewed-896-cosine/independent-audit.json),
and [configuration](outputs/autonomous-20260913/results/auto-20260913-002-reviewed-896-cosine/train/config.json).

The next active experiment uses the verified surgical DINO ResNet50 backbone
as job `8719856880257597440`. Its queued generic-pretraining control retains
the same optimization settings. Further queued comparisons test Lovasz,
removal of auxiliary loss during matched fine-tuning, a full-rate detail
backbone and matched standard decoder, exact-native input, and lower initial
learning rate. These remain hypotheses until their runs complete.

**TRAIN-only diagnostic, 20:34 UTC:** Six two-image batches from 12 distinct
original TRAIN cases were checked at run002's selected epoch-43 checkpoint.
Image/mask bytes were verified against the same immutable cloud data archive.
With the existing 0.4 auxiliary coefficient, the auxiliary/main gradient norm
ratio in shared early backbone layers had median **18.17×** (range
7.17–43.10×), and gradient cosine was negative in **3 of 6** batches.
The whole-backbone median ratio was 9.84×. This supports testing removal of
auxiliary loss; it does not establish a validation gain or describe AdamW's
moment-adjusted update direction. The diagnostic used CPU float32 and local
library versions, fixed-seed training dropout, frozen BatchNorm, and no
optimizer updates. Frames containing ignored pixels were excluded from this
small diagnostic sample; training data and all experiment splits remain intact.
Evidence: [diagnostic settings, identities and measurements](outputs/autonomous-20260913/diagnostics/gradient_aux_run002.json)
and [reproduction script](outputs/autonomous-20260913/diagnostics/gradient_aux_run002.py).

**Third result, verified 20:37 UTC:** `auto-20260913-003-detail-672-cosine`
completed 80 epochs and selected epoch **26**. Native foreground IoU was
**39.7987%**, small pooled IoU **19.1657%**, and small equal-case IoU
**18.8473%**. The new stride-four decoder with a 0.1 backbone learning-rate
multiplier underperformed the reviewed baseline. This does not isolate the
decoder from the reduced backbone rate; the queued full-rate decoder and
matched standard MobileNet control will test that distinction.

Vertex job: `586918903179902976`. Selected checkpoint SHA-256:
`f2b5940dcb4370cc55ca0920b78869041a459fe2152d8e4791b0adbfab51f7e2`.
Evidence: [native metrics](outputs/autonomous-20260913/results/auto-20260913-003-detail-672-cosine/train/metrics-val-original.json),
[independent audit](outputs/autonomous-20260913/results/auto-20260913-003-detail-672-cosine/independent-audit.json),
and [configuration](outputs/autonomous-20260913/results/auto-20260913-003-detail-672-cosine/train/config.json).
The generic ResNet50 control was submitted as job `6078495698804801536`.

**Fourth result, verified 20:53 UTC:** `auto-20260913-004-surgical-r50-cosine`
completed 60 epochs and selected epoch **21**. Native foreground IoU reached
**49.2846%**, small pooled IoU **30.7720%**, and small equal-case IoU
**28.3335%**. Relative to the reviewed seed-42 MobileNet baseline, these are
gains of **3.413**, **3.118**, and **1.905 pp**, respectively.

Five foreground classes improved: duct +5.32 pp, triangle +5.59, tools +4.62,
gallbladder +3.39, and plate +2.53. Artery IoU fell 0.97 pp. Eight of ten cases
improved foreground IoU, but case 137 lost 5.58 pp foreground and 11.94 pp
small-anatomy IoU; case 144 also regressed. This becomes the search leader,
without changing the demo checkpoint. The still-running matched generic
ResNet50 control is required to separate surgical pretraining from the shared
architecture and optimization changes.

Vertex job: `8719856880257597440`. Selected checkpoint SHA-256:
`0983cb099b806b1a6c07c17ac5234b4ef9a198a878b44a88da9b475e823a2f98`.
Evidence: [native metrics](outputs/autonomous-20260913/results/auto-20260913-004-surgical-r50-cosine/train/metrics-val-original.json),
[independent audit](outputs/autonomous-20260913/results/auto-20260913-004-surgical-r50-cosine/independent-audit.json),
and [configuration](outputs/autonomous-20260913/results/auto-20260913-004-surgical-r50-cosine/train/config.json).
The matched MoCo-v2 pretraining run was submitted next as job
`3152281840920821760`, with an independently verified 318-tensor backbone
wrapper and unchanged 60-epoch settings.

**Fifth result, verified 21:06 UTC:** `auto-20260913-005-generic-r50-matched`
completed 60 epochs and selected epoch **24**. Native foreground IoU reached
**50.0944%**, small pooled IoU **32.0247%**, and small equal-case IoU
**28.5296%**. All six pooled foreground classes improved over the reviewed
MobileNet baseline. This is the new search leader.

Against its matched DINO ResNet50 arm, the gains are 0.810 pp foreground,
1.253 pp small pooled, and 0.196 pp small equal-case IoU. Artery improved
4.280 pp and plate 2.719 pp, while triangle fell 2.352 pp. Five cases improved
and five declined. The result supports the shared ResNet50 optimization
recipe; it does not establish a benefit from surgical pretraining. A fresh
generic ResNet50 trial at 896 × 512 was moved to the front of the future queue,
with the same 60 epochs, warmup, learning rates, loss, seed and data.

Vertex job: `6078495698804801536`. Selected checkpoint SHA-256:
`0c06f4941a4a4e09c270a566b6127d3ee1065da85ad7506d59d6398da7e73c28`.
Evidence: [native metrics](outputs/autonomous-20260913/results/auto-20260913-005-generic-r50-matched/train/metrics-val-original.json),
[independent audit](outputs/autonomous-20260913/results/auto-20260913-005-generic-r50-matched/independent-audit.json),
and [configuration](outputs/autonomous-20260913/results/auto-20260913-005-generic-r50-matched/train/config.json).
The MobileNet Lovasz comparison was submitted as job `6600209568138002432`;
the matched main-only continuation remains queued.

**Sixth result, verified 21:22 UTC:** `auto-20260913-006-moco-r50-matched`
completed 60 epochs and selected epoch **32**. Native foreground IoU reached
**51.0913%**, small pooled IoU **33.2681%**, and small equal-case IoU
**28.3267%**. Against the matched generic ResNet50 arm, foreground and small
pooled IoU improve 0.997 and 1.243 pp, while small equal-case IoU declines
0.203 pp. Against DINO, the corresponding differences are +1.807, +2.496,
and -0.007 pp. This is the pooled-score leader; it is not a demonstrated
improvement across cases or random seeds.

Vertex job: `3152281840920821760`. Selected checkpoint SHA-256:
`1bc74c1f7915c01370ec38d9c6450f5a82c0cf8b558cb76f919f43cf9e04fd1d`.
Evidence: [native metrics](outputs/autonomous-20260913/results/auto-20260913-006-moco-r50-matched/train/metrics-val-original.json),
[independent audit](outputs/autonomous-20260913/results/auto-20260913-006-moco-r50-matched/independent-audit.json),
and [configuration](outputs/autonomous-20260913/results/auto-20260913-006-moco-r50-matched/train/config.json).
The generic 896 × 512 control started as job `8922378126000914432`. A matched
fresh MoCo 896 × 512 trial was added next, preserving initialization, 60-epoch
schedule, learning rates, loss, seed and data.

**Seventh result, verified 21:24 UTC:** `auto-20260913-007-warm-lovasz`
completed 40 epochs and selected epoch **1**. Native foreground IoU was
**45.6471%**, small pooled IoU **27.0547%**, and small equal-case IoU
**25.1279%**. Relative to matched CE continuation run001, all three declined
(-0.229, -0.460 and -0.735 pp). Adding 0.25 main-head Lovasz during gentle
fine-tuning did not help; later epochs degraded further.

Vertex job: `6600209568138002432`. Selected checkpoint SHA-256:
`290ff300c32df8d93db348fe0f68cfca0ff8c54eebf3a5d56fba81b1ef6165f3`.
Evidence: [native metrics](outputs/autonomous-20260913/results/auto-20260913-007-warm-lovasz/train/metrics-val-original.json),
[independent audit](outputs/autonomous-20260913/results/auto-20260913-007-warm-lovasz/independent-audit.json),
and [configuration](outputs/autonomous-20260913/results/auto-20260913-007-warm-lovasz/train/config.json).
The matched main-only continuation started as job `3536072971665801216`.

**Queue decision, 21:25 UTC:** Prioritize the two stronger fresh ResNet50
resolution arms, then separate MoCo mild-augmentation and fresh Lovasz trials.
The latter changes only the objective relative to run006; the negative warm
MobileNet result is not evidence of a benefit in the new setting. All three
672-pixel ResNet50 runs increasingly overfit after selected epochs 21/24/32,
supporting a bounded regularization test. Their auxiliary term contributes
about 30–31% of the selected-epoch training objective, unlike the large
MobileNet imbalance, so removing ResNet50 auxiliary loss is lower priority.
The full-rate detail decoder and its matched standard MobileNet control remain
queued. Fixed validation/test identities and source/data hashes are preserved.

At 21:31 UTC, the lowest-priority low-rate MobileNet trial was replaced with
a seed-43 repeat of the current MoCo leader. The selected recipe may be revised
if the pending controlled comparisons produce a stronger result. This checks
training-trajectory variation before adding another regularization parameter;
one additional seed alone cannot establish significance. A local controller
reliability review also added deadline-bounded artifact collection and a final
audit pass after owned compute terminates. The supervisor resumed at 21:28 UTC
with the same deadline and jobs; all 306 ML tests passed. Worker source remains
version 4 for current jobs.

**Run009 result, verified 21:39 UTC:** `auto-20260913-009-warm-main-only`
completed 40 epochs and selected epoch **1**. Native foreground IoU was
**45.9477%**, small pooled IoU **27.5810%**, and small equal-case IoU
**25.8473%**. Against matched CE continuation run001, these differ by only
+0.072, +0.066, and -0.016 pp. Against the original reviewed checkpoint,
foreground IoU is +0.076 pp, while both small-anatomy metrics decline
(-0.074 pp pooled and -0.581 pp equal-case).

Removing auxiliary loss does not rescue continued MobileNet fine-tuning.
All three warm arms select epoch 1 and decline afterward. The measured
auxiliary-gradient imbalance therefore does not by itself explain the plateau;
additional warm auxiliary-loss trials are lower priority.

Vertex job: `3536072971665801216`. Selected checkpoint SHA-256:
`0097868e850392092f5145fe90348ffd4207fdfdb36b702cf73c6d4d3857b2b0`.
Evidence: [native metrics](outputs/autonomous-20260913/results/auto-20260913-009-warm-main-only/train/metrics-val-original.json),
[independent audit](outputs/autonomous-20260913/results/auto-20260913-009-warm-main-only/independent-audit.json),
and [configuration](outputs/autonomous-20260913/results/auto-20260913-009-warm-main-only/train/config.json).
Run008 generic ResNet50 at 896 × 512 remains active. Run010's matched MoCo
resolution experiment was submitted as job `974650683598831616`.

**Run008 result, verified 21:59 UTC:** `auto-20260913-008-generic-r50-896`
completed its full 60-epoch schedule and selected epoch **34**. Native
foreground IoU was **49.5975%**, small pooled IoU **31.3271%**, and small
equal-case IoU **28.3397%**. Compared with matched 672 × 384 generic ResNet50
run005, all three decline: -0.497, -0.698 and -0.190 pp, respectively.
Changing only the grid to 896 × 512 did not improve this recipe.
Triangle IoU improved 4.278 pp, but artery fell 2.781 pp, plate 3.374 pp,
and duct 0.914 pp; gallbladder and tools changed little. Median epoch time
increased from 22.59 to 31.36 seconds, about 39%. Larger generic input is
therefore lower priority than the remaining objective and regularization tests.

Vertex job: `8922378126000914432`. Selected checkpoint SHA-256:
`3c83e122d8abd365ec54652f865feddff32bf3a16040cc596a29dc98d146ee31`.
Evidence: [native metrics](outputs/autonomous-20260913/results/auto-20260913-008-generic-r50-896/train/metrics-val-original.json),
[independent audit](outputs/autonomous-20260913/results/auto-20260913-008-generic-r50-896/independent-audit.json),
and [configuration](outputs/autonomous-20260913/results/auto-20260913-008-generic-r50-896/train/config.json).
The matched MoCo resolution arm remains active. Run011 now tests mild paired
flips and brightness/contrast augmentation with the fresh 672 × 384 MoCo
recipe as job `8931103850278944768`.

**Remaining-window decision, 22:05 UTC:** Replace the lowest-priority exact-native
MobileNet trial with a seed-44 repeat of the winning recipe and move both seed
repeats ahead of the matched standard-MobileNet decoder control. The next
priorities are fresh MoCo Lovasz, full-rate detail decoder, winner seed43,
winner seed44, then the standard decoder control. Repeat configurations remain
subject to the pending objective/augmentation results before submission.
Actual observed 60-epoch 672-pixel ResNet50 jobs take about 27 minutes from
submission to termination, so earlier placement protects full schedules within
the fixed deadline. Any reduced or interrupted comparison will be identified
explicitly. No further increase beyond 896-pixel input is planned.

**Run010 result, verified 22:18 UTC:** `auto-20260913-010-moco-r50-896`
completed 60 epochs and selected epoch **38**. Native foreground IoU was
**50.6326%**, small pooled IoU **32.5272%**, and small equal-case IoU
**28.3893%**. Relative to matched 672 × 384 MoCo run006, foreground and small
pooled IoU decline 0.459 and 0.741 pp; equal-case small IoU improves only
0.063 pp. Both matched ResNet50 resolution increases fail to improve the
primary metric, so the smaller MoCo recipe remains the research leader.
The larger MoCo grid improves artery IoU 2.189 pp but loses 5.033 pp on plate.
Five cases improve and five decline. Median epoch time increases from 22.22
to 32.18 seconds, about 45%, making the smaller recipe preferable for repeats.

Vertex job: `974650683598831616`. Selected checkpoint SHA-256:
`5fedebd2367e9709784dccb1b479256d488d758061f2acdc718c2ba6a2cbb67f`.
Evidence: [native metrics](outputs/autonomous-20260913/results/auto-20260913-010-moco-r50-896/train/metrics-val-original.json),
[independent audit](outputs/autonomous-20260913/results/auto-20260913-010-moco-r50-896/independent-audit.json),
and [configuration](outputs/autonomous-20260913/results/auto-20260913-010-moco-r50-896/train/config.json).
Run012 tests fresh MoCo training with CE plus 0.25 main-head Lovasz as job
`5149206063194570752`; the augmentation arm remains active.

At 22:21 UTC, the local supervisor gained validated adaptive seed-pair markers.
The first repeat now selects the best fully completed, independently audited
owned run at launch and persists that full recipe for both seeds 43 and 44.
This avoids a race between a newly collected winner and immediate submission.
The pair preserves the source recipe even if a later run improves; actual
source identity and resolved settings are saved with each job. Runtime limits
remain unchanged, and incomplete schedules cannot supply a "full" source.
All 312 ML tests passed. The supervisor resumed with the original deadline,
owned jobs and worker source v4 unchanged.

**Run011 result, verified 22:31 UTC:** `auto-20260913-011-moco-r50-mild`
completed 60 epochs and selected epoch **43**. Native foreground IoU was
**50.8060%**, small pooled IoU **32.9555%**, and small equal-case IoU
**27.7801%**. Compared with matched unaugmented MoCo run006, all three decline
(-0.285, -0.313 and -0.547 pp). Mild augmentation does not improve this
single-seed comparison, so run006 remains the leader pending the fresh Lovasz
and full-rate decoder results.

Vertex job: `8931103850278944768`. Selected checkpoint SHA-256:
`40a92ad2d4dcbe3b66aa282ce22edf8f37a36dfc4f580b6056d3fbe433e5c563`.
Evidence: [native metrics](outputs/autonomous-20260913/results/auto-20260913-011-moco-r50-mild/train/metrics-val-original.json),
[independent audit](outputs/autonomous-20260913/results/auto-20260913-011-moco-r50-mild/independent-audit.json),
and [configuration](outputs/autonomous-20260913/results/auto-20260913-011-moco-r50-mild/train/config.json).
Run013 tests the detail decoder with backbone multiplier 1.0, matching run003's
80-epoch schedule and other settings, as job `1719151996998516736`.

**Run012 result, collected and verified 23:00 UTC:**
`auto-20260913-012-moco-r50-lovasz` completed 60 epochs and selected epoch
**34**. Native foreground IoU reached **52.0409%**, small pooled IoU
**34.6151%**, and small equal-case IoU **28.2247%**. Relative to matched CE
MoCo run006, foreground and small pooled IoU improve **0.950** and **1.347 pp**,
while small equal-case IoU declines 0.102 pp. This becomes the primary-score
leader; case-equal improvement is not demonstrated.

Vertex job: `5149206063194570752`. Selected checkpoint SHA-256:
`b037e85dff5f0ca552258d733cd02364717f99b57d93e6a0b10bbcd3c2ceea0f`.
Evidence: [native metrics](outputs/autonomous-20260913/results/auto-20260913-012-moco-r50-lovasz/train/metrics-val-original.json),
[independent audit](outputs/autonomous-20260913/results/auto-20260913-012-moco-r50-lovasz/independent-audit.json),
and [configuration](outputs/autonomous-20260913/results/auto-20260913-012-moco-r50-lovasz/train/config.json).

**Run013 result, collected and verified 23:09 UTC:**
`auto-20260913-013-detail-full-backbone` completed 80 epochs and selected epoch
**28**. Native foreground IoU was **45.7728%**, small pooled IoU **27.6095%**,
and small equal-case IoU **22.2854%**. Raising the backbone multiplier from
0.1 to 1.0 improves on matched detail run003 by 5.974 pp foreground, but the
result remains below the ResNet50 candidates. The separately planned standard
MobileNet architecture control is deferred because its full 80-epoch schedule
will no longer fit after the two higher-priority seed repeats.

Vertex job: `1719151996998516736`. Selected checkpoint SHA-256:
`c884bacba21c6ec5d7a3714613492f10e59955f5bd4fa897ab37ac84aba22199`.
Evidence: [native metrics](outputs/autonomous-20260913/results/auto-20260913-013-detail-full-backbone/train/metrics-val-original.json),
[independent audit](outputs/autonomous-20260913/results/auto-20260913-013-detail-full-backbone/independent-audit.json),
and [configuration](outputs/autonomous-20260913/results/auto-20260913-013-detail-full-backbone/train/config.json).

**Execution timing and final repeats:** Cloud run012 ended at 22:45:35 UTC,
but local event updates and collection were delayed; its audit completed at
23:00:25, followed by run013 at 23:09:35. The local supervisor was alive and
collecting artifacts when inspected. The cause of the delay was not established;
the authorization deadline was not extended. At 23:09:36–37, it automatically
froze run012's full recipe for group `leader-seeds-43-44` and submitted seeds
43 and 44 as jobs `6283761325571047424` and `5101566423386292224`. Each keeps
60 epochs and every training setting except seed. The existing duration cap
still applies; final reports must distinguish complete repeats from partial
ones. The final unmatched short MobileNet control was removed from the queue.
The bounded controller may attempt a short leader refinement only if sufficient
time remains after both primary repeat slots progress.

**Final repeats, independently verified 23:40 UTC:** Both repeats completed
their full 60-epoch schedules despite the deadline duration caps. Their saved
model, optimizer, data, loss, sampling and augmentation settings match run012;
only seed and execution metadata differ. No further trial fits the eight-minute
launch cutoff.

| Run / seed | Selected epoch | Native foreground IoU | Small pooled IoU | Small equal-case IoU |
| --- | ---: | ---: | ---: | ---: |
| 012 / 42 | 34/60 | 52.0409% | 34.6151% | 28.2247% |
| 014 / 43 | 24/60 | 51.6340% | 34.0068% | 30.0329% |
| 015 / 44 | 30/60 | 50.8585% | 32.6795% | 28.4110% |
| Three-seed mean | — | 51.5111% | 33.7672% | 28.8895% |

Run014 job `6283761325571047424` finished at 23:37:54 UTC; checkpoint SHA-256
`5d1c98b979b7d03c32b6ae34d024f09de138ef616130bce6519efa367e8e5a3c`.
Run015 job `5101566423386292224` finished at 23:38:17 UTC; checkpoint SHA-256
`78faddac3608296a21edc482ef3922f647cdf43fbfdd0bd610252a57643d807c`.
The foreground range is 50.8585–52.0409%. This checks variability of the
adaptively selected Lovasz recipe; it is not a paired multi-seed CE comparison
and does not establish statistical significance. See the
[independent replication report](outputs/autonomous-20260913/reports/replication-results.md)
for full config comparisons and artifact hashes.

**Search outcome:** All **15 new Vertex jobs succeeded**, completing **900
epochs** in total. Every requested schedule completed; there were no partial,
failed or unaudited runs. A fresh cloud query at 23:40 UTC confirmed all 15
owned jobs were terminal. The primary leader is run012, MoCo-initialized
DeepLabV3–ResNet50 at 672 × 384 with balanced CE + 0.25 main-head Lovasz,
cosine scheduling and a 0.1 backbone learning-rate multiplier. It improves
the reviewed seed-42 reference from **45.8717% to 52.0409%** native foreground
IoU (**+6.1692 percentage points**), below the **75% target**. Pooled
small-anatomy IoU is 34.6151%, but its 28.2247% equal-case small IoU is
essentially unchanged from the matched MoCo CE run's 28.3267%.

The search selected repeatedly on the same 75 validation frames from ten cases.
No test evaluation was run and no checkpoint was automatically promoted to the
demo. Small structures remain the main limitation, particularly artery and
plate. The next evidence-backed step is to obtain human review of the separate
50-image TRAIN-only proposal batch, then compare the frozen leading recipe
with and without the approved additions across matched seeds. Unreviewed
proposals must remain excluded; adding labels is a hypothesis, not a guarantee
of reaching 75%. The standard MobileNet decoder control remains deferred.
The current local ML suite passes **312 tests**.

**Deadline closure, 23:44:23 UTC:** The controller recorded `deadline_reached`
and stopped successfully at the fixed four-hour deadline. All 15 owned jobs
were already terminal and independently audited; no cancellation or additional
trial was needed. The final snapshot records **15 verified runs, zero active
jobs, zero audit failures, and an unmet 75% target**. The controller and its
temporary keep-awake process have exited.

Final evidence: [results report](outputs/autonomous-20260913/reports/final/RESULTS.md),
[verified summary](outputs/autonomous-20260913/reports/final/summary.json),
[three-seed comparison](outputs/autonomous-20260913/reports/replication-results.md),
and [results chart](outputs/autonomous-20260913/diagnostics/plots-final/training-results.png).


## September 14, 2026 — returned review batch002 and bounded retraining

The lead supplied the second review export and explicitly requested training,
promotion of the best local model, and logging. Source review SHA-256
`f65147d81824c0610af5fb73150d2ffa2c9c3b8bd7f457825b5f4afe6fa56214`
passed strict validation against bundle
`91ad669b5a0b5225468908e25a1b344808571234ad3a36a928798af54e419586`.
The original 141 decisions are preserved: 73 accepted, 48 edited, 19 rejected,
one needs-expert. Two accepted masks had contradictory notes; the lead
confirmed excluding both for this run. A separate resolution now records those
exclusions without changing the review or its 121-mask import receipt.

**Prepared supervision:** 119 training-eligible masks on 46 new images from 25
TRAIN cases; 633,893 retained native foreground pixels and 16,216 conflict pixels
ignored. The combined manifest contains 407 train / 75 val / 74 test images,
preserving the original data and first pilot. Unknown/conflicting pixels stay
255; same-class positives are unioned. Four no-supervision images are omitted.
Localized plate overlap losses are recorded in the preparation report.

**Frozen hypothesis and budget:** extra reviewed small-anatomy supervision may
improve the leading surgical-MoCo ResNet50 recipe. Compare fresh seeds 42 and 43
at 672 × 384, CE + 0.25 main Lovasz, auxiliary weight 0.4, batch 2, LR 0.0003,
cosine schedule with three-epoch warmup, backbone multiplier 0.1, no augmentation,
uniform sampling, and 53 epochs. This is 10,812 optimizer updates versus 10,860
in the historical 60-epoch controls, within 0.44%; warmup length in updates,
selection opportunities, and class weights differ and will be reported.
At most two A100 jobs may launch in a new two-hour capped state; stop early once
both runs are terminal and their results are audited. The old four-hour state
remains closed. No test evaluation is planned.

**Engineering:** target preparation now supports attributed training-only
candidate exclusions; the controller supports optional closure after exhausting
a finite launch budget. Original review bytes and historical behavior remain
intact. The full local ML suite passes 323 tests. A separately verified immutable
source archive and prepared-data package will be bound to each new launch.

**Promotion rule:** compare every candidate with the verified existing run012
leader (52.040885698% native foreground IoU; SHA-256
`b037e85dff5f0ca552258d733cd02364717f99b57d93e6a0b10bbcd3c2ceea0f`).
Publish the best checkpoint under `ml/weights/current/best.pt` with a selection
receipt after evaluation; preserve old checkpoints and prediction identities.
See [batch002 record](REVIEW_BATCH_002.md) and
[event log](outputs/reviewed-batch002-20260914/events.jsonl).


**Pre-launch verification and local promotion:** All 46 targets match independent
RLE reconstruction and the actual 672 × 384 loader. All 1,020 inherited files,
the original ordered sample prefix, and validation/test fingerprints match.
Cloud staging verified 1,200 bound files; all 19 training/evaluation source
files match the historical control archive. The extracted source suite also
passes all 323 tests. Run012 was copied to `ml/weights/current/best.pt`, its
checksum verified, and strict CPU loading plus one-frame inference and contract
validation passed. The selection receipt preserves source identity and metrics.

**Cloud execution pending:** Automatic approval review rejected the source
upload because it requires explicit user authorization to export project and
review materials to the private Google Cloud bucket. Approval for both concrete
bundles and the two bounded A100 runs was requested. No new upload or training
job has succeeded; training is not represented as complete. The frozen launch
plan, source/data hashes, independent audits and event log are ready. The new
UTC window will begin only with the approved launch.


**Approved launch, September 14 04:25 UTC:** The user explicitly approved the
previously described uploads and two bounded A100 runs. Both immutable bundles
uploaded; cloud MD5/size matches were checked against local bytes. A fresh
controller state fixes the window at 04:25:44–06:25:44 UTC, max two launches and
two workers, with early closure after terminal result audits. Both 53-epoch
recipes retain their frozen source/data/backbone identities. Submitted jobs:

- Seed 42: `561146350624833536`, `review2-20260914-001-batch002-moco-lovasz-s42`.

- Seed 43: `4395961433330810880`, `review2-20260914-002-batch002-moco-lovasz-s43`.

Initial state is pending worker startup; submission does not establish training
completion or improved metrics. See the live event log and controller state.

**Training observed, 04:29 UTC:** Both workers completed epoch 1 training batches
(204/204), confirming that the 407-image prepared dataset reached optimization.
Live input-grid validation is kept separate from final native-grid selection.


**Completed batch, September 14 at 05:11 UTC; promotion at 05:16 UTC:** Both jobs
succeeded with all 53 requested epochs (106 total), no duration truncation and
no additional launches. Worker completion times were 04:53:20/04:53:32 UTC;
Vertex terminal times were 04:53:37/04:53:56 UTC. Local collection completed later,
at 05:10:30/05:11:20 UTC; this collection/inspection delay is recorded without
claiming an established cause. The controller then closed as
`experiment_batch_complete` at 05:11:20 UTC, before the fixed 06:25:44 deadline.
A fresh cloud query confirmed both jobs terminal.

| Seed | Selected/completed epoch | Native foreground IoU | Small pooled IoU | Small equal-case IoU | Checkpoint SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| 42 | 34/53 | 52.825114156% | 35.705296118% | 32.275247233% | `b406ed42ab0394edba22e1dde0edc2865a6346adb61c4bea7ab0bc00d08e1911` |
| 43 | 39/53 | 51.917237681% | 34.373667789% | 31.421636817% | `13db3a323cf56fd4f28bb79f710d3294ea5c396b4d762023c0dfee5c4276285b` |

All 36 collected artifacts verified against their byte counts and SHA-256.
Both native validation audits recomputed confusion metrics, verified fixed
75-frame/ten-case identities, ignored-pixel supports, class mapping and exact
checkpoint/configuration identity. The paired helper independently re-audited
both new runs and historical controls, including input-file fingerprints.
Both seeds improve all three headline metrics. Mean paired improvements are
+0.5338 foreground, +0.7285 small-pooled and +2.7196 equal-case percentage points.
Actual optimizer updates are 10,812 per new run versus 10,860 per historical run;
class weights, sampling order, warmup updates and selection opportunities differ.
These are historical matched-seed controls, not a new fully identical-schedule
trial; no significance or clinical-validity claim is made.

Seed 42 is promoted under `ml/weights/current/best.pt`: 52.8251% versus the previous
52.0409% (+0.7842 points). Strict CPU loading and copied-file checksum checks
passed. The previous selection receipt and original checkpoint are retained.
Duct IoU gains 4.424 points, artery 4.209, plate 2.714; triangle-dissection loses 6.987,
with precision down 15.821 points and recall up 17.577. Small-anatomy case IoU
improves in 7/10 cases; the worst decline is case 131 (-2.688 points). Seed 43 improves
6/10 cases, with its largest decline in case 146 (-6.826 points). Case means
exclude classes absent from both prediction and truth; the number of scored
classes can differ, notably for case 137. These limitations remain explicit
despite the aggregate win. The separate [result audit](outputs/reviewed-batch002-20260914/independent-data-audit/result-audit.json)
also confirms both complete schedules, checkpoint metadata and native confusion arithmetic.

[Paired report](outputs/reviewed-batch002-20260914/reports/paired-final/RESULTS.md),
[summary](outputs/reviewed-batch002-20260914/reports/paired-final/summary.json),
[comparison chart](outputs/reviewed-batch002-20260914/reports/paired-final/comparison.png),
and [promoted selection](outputs/reviewed-batch002-20260914/promoted-selection.json)
record details. The chart was visually checked. Matplotlib dependencies were
installed only under the ignored report-tools directory; the training and
inference environment was not changed. The next evidence-backed labeling focus
is consistency of triangle-dissection boundaries and its false-positive regions;
no further experiment is launched in this completed budget.

**Refreshed video handoff, September 14 at 05:29 UTC:** Local inference with the
promoted checkpoint completed the full Gupta clip at threshold 0.5. The export
ran from 05:16:32 to 05:29:23 UTC and contains all 1,056 frames, with no frame
limit or partial output. Python contract validation passed. Independent decoding
verified every exact presentation timestamp, frame number, original 1280 × 720
dimension and all 1,056 raw masks. The actual frontend parser and matcher accept
every result; the 10-second frame (240) displays ten predicted components with
no matching-result warning. Unrelated media and reviewed-annotation source
selections correctly return no match.

The [new predictions.json](outputs/reviewed-batch002-20260914/gupta-current/predictions.json)
is 71,053,910 bytes, SHA-256
`d38c2788b109fe057e5734d770235fa014c1636aae39ac78b94b38300af14d8b`.
It records model version `2026-09-14T04:29:29.757933Z-epoch-34`; the input clip
retains SHA-256 `3bbc15cca1d23915fb79700aa28105cbb41989744836c434c1a2e74f59f7a927`.
The previous run012 export remains intact. The event log stores the exact
inference command; [identity verification](outputs/reviewed-batch002-20260914/gupta-current/identity-verification.json),
[frontend verification](outputs/reviewed-batch002-20260914/gupta-current/frontend-verification.json)
and [schema output](outputs/reviewed-batch002-20260914/gupta-current/schema-validation.log)
record the checks. These establish file and display compatibility, not anatomical
accuracy on this public clip. Current model, status, review-batch and demo-media
documentation now point to the completed handoff. Both cloud jobs and the
controller are stopped; no further training was launched. The batch's best
verified foreground IoU remains 52.8251%, below the 75% target.
