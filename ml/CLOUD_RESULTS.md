# Cloud results and comparability audit

Snapshot: **September 13, 2026, 04:10 UTC**. All five jobs in the
[frozen plan](CLOUD_EXPERIMENTS.md) reached Vertex `JOB_STATE_SUCCEEDED`.
All requested epochs and original-grid validation completed, and all five
artifact sets were collected with hash verification. The selected demo model
remains **`small-004-resolution/best.pt`**. The completed comparison retains it
as the default; `cloud-002-longer` is the best cloud research candidate and is
not promoted to the demo.

## Results on the same validation frames

Every score below uses the same **75 frames / 10 cases**, with unchanged
**854 × 480** masks. Logits resize to the original grid before argmax; no
confidence or polygon filter is applied. Small-anatomy means cover cystic duct,
cystic artery, cystic plate, and hepatocystic triangle dissection. Six-class
foreground means also include gallbladder and tool; background is excluded.

| Run | Device | Epochs completed / selected | Small pooled IoU | Small case-equal IoU | Six-class pooled IoU | Epoch seconds: total / mean |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| [Local reference](outputs/small-004-resolution/metrics-val-original.json) | Apple MPS | 12 / 9 | 24.36% | 25.73% | 42.94% | 410.2 / 34.18 |
| [001: CUDA matched control](outputs/cloud-001-control/train/metrics-val-original.json) | NVIDIA A100 | 12 / 7 | 23.53% | 21.89% | 42.01% | 199.8 / 16.65 |
| [002: Longer control](outputs/cloud-002-longer/train/metrics-val-original.json) | NVIDIA A100 | 40 / 38 | **26.36%** | **25.35%** | **44.77%** | 671.4 / 16.79 |
| [003: Mild augmentation](outputs/cloud-003-augmentation/train/metrics-val-original.json) | NVIDIA A100 | 40 / 32 | 24.79% | 20.72% | 43.83% | 757.6 / 18.94 |
| [004: Case-balanced sampling](outputs/cloud-004-case-balanced/train/metrics-val-original.json) | NVIDIA A100 | 40 / 38 | 23.95% | 22.04% | 43.19% | 673.9 / 16.85 |
| [005: ResNet50](outputs/cloud-005-resnet50/train/metrics-val-original.json) | NVIDIA A100 | 40 / 32 | 25.46% | 23.38% | 44.26% | 866.6 / 21.66 |

Bold identifies the highest **cloud** score in each aggregate column. The
40-epoch MobileNet control ranks highest among the five cloud runs on all
three aggregates. Compared with the 12-epoch CUDA run, it scores 2.83 percentage
points higher on pooled small-anatomy IoU, 3.47 points higher case-equal, and
2.76 points higher on six-class IoU. None of the three independent changes
improves these aggregate measures over the 40-epoch control in this experiment.

The longer control's case-equal small-anatomy IoU remains slightly below the
local reference (25.35% versus 25.73%), and artery recall remains lower
(24.63% versus 34.27%). Higher pooled means therefore do not establish an
unqualified improvement over the demo model. Each of the three alternatives
also falls below the local model on case-equal small-anatomy IoU and artery
recall. No cloud candidate is promoted, and no additional jobs are launched.

Epoch durations sum/average the recorded training-plus-validation epoch timers.
They exclude provisioning, setup, downloads, final original-grid evaluation,
and artifact transfer. They are not total job durations or an isolated GPU
inference benchmark.

## Per-class tradeoffs

All entries in the following table are pooled original-grid **IoU percentages**.

| Class | Local | 001: 12 epochs | 002: 40 epochs | 003: Augmentation | 004: Case balancing | 005: ResNet50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gallbladder | 81.28 | 79.77 | 81.30 | 80.80 | 80.72 | 80.21 |
| Cystic duct | 40.68 | 44.83 | 43.45 | 33.40 | 38.41 | 41.03 |
| Cystic artery | 16.06 | 12.31 | 15.75 | 13.43 | 10.26 | 17.62 |
| Cystic plate | 19.71 | 10.00 | 18.28 | 21.90 | 19.19 | 13.47 |
| Triangle dissection | 20.97 | 26.99 | 27.96 | 30.44 | 27.92 | 29.73 |
| Tool | 78.91 | 78.13 | 81.88 | 83.01 | 82.63 | 83.50 |

Small-structure **recall percentages** show which apparent gains accompany more
missed pixels; complete precision and Dice measures remain in each linked report.

| Class | Local | 001: 12 epochs | 002: 40 epochs | 003: Augmentation | 004: Case balancing | 005: ResNet50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cystic duct | 51.78 | 64.15 | 59.32 | 40.17 | 49.18 | 49.86 |
| Cystic artery | 34.27 | 24.61 | 24.63 | 20.78 | 19.33 | 29.68 |
| Cystic plate | 33.37 | 15.01 | 30.41 | 31.47 | 37.23 | 25.23 |
| Triangle dissection | 60.26 | 62.86 | 63.85 | 52.88 | 53.23 | 48.12 |

- **Longer control:** artery precision rises from 19.76% to 30.39% versus 001
  while recall stays near 24.6%; this does not show materially more artery
  pixels detected. Plate recall recovers, but duct recall decreases.
- **Augmentation:** plate and triangle IoU improve versus 002, while duct and
  artery IoU/recall decrease. Triangle IoU rises despite recall falling from
  63.85% to 52.88%, alongside precision increasing from 33.21% to 41.77%.
- **Case balancing:** plate recall improves to 37.23%, but duct, artery, and
  triangle recall decrease versus 002. Artery IoU falls to 10.26%.
- **ResNet50:** artery IoU/recall improve over 002, but plate IoU/recall and
  duct/triangle recall decrease. Triangle precision rises to 43.75% while its
  recall falls to 48.12%. Its larger backbone does not win the aggregate scores.

These are single-seed validation observations. The 12- and 40-epoch CUDA
controls already follow different trajectories during their first 12 epochs:
first-epoch resized validation IoU is 33.55% versus 32.81%. The code seeds random
sources but does not require deterministic CUDA algorithms. A shared seed is
not deterministic causal proof that the changed setting alone explains a
between-run result; repeated runs would be needed to estimate variability.

## Independent comparability audit

**All cloud runs:** complete manifests and ordered evaluation frame records are
identical. Metric policies, normalization, class weights and their pixel counts,
source-255 ignore settings, quarantined frame, and software versions match.
All five job configurations reference the same source and data bundle hashes.
Every collected artifact's SHA-256 and byte count matches its completion record.

The four 40-epoch configurations were compared directly. Relative to 002, their
only saved config differences, apart from run identity, are:

| Candidate | Actual configuration differences |
| --- | --- |
| 003 | `augmentation=none → mild` and the corresponding transform metadata |
| 004 | `sampling=uniform → case_balanced` and replacement/weight-formula metadata |
| 005 | `architecture` and corresponding `model_id` change to ResNet50 |

For 001 versus 002, only `epochs` and `run_id` differ. Every cloud run uses
672 × 384, batch 2, learning rate 0.0003, seed 42, frozen BatchNorm, and the same
balanced-loss calculation. Class weights count each unaugmented selected train
mask once, including for replacement sampling. All runs share the same 343
training frames / 30 cases and 75 validation frames / 10 cases; case-balanced
training draws with replacement rather than visiting each frame once per epoch.

**Local versus cloud:** after replacing absolute dataset roots with a common
placeholder, complete manifests and all 75 evaluation frame records are
identical. SHA-256 comparisons of all **836 train/validation image and mask
files** against the submitted data bundle found zero differences. Seven-channel
class maps, ignore policy, and `153_32700` exclusion match. Exact weight arrays
and counts match: 87,976,779 scored and 533,685 ignored resized training pixels.
Original-grid validation scores 30,683,409 pixels and ignores 60,591 in all runs.
Cloud config explicitly records default augmentation/sampling fields absent from
the earlier local implementation; its behavior was no augmentation/uniform.

Each run selects the first highest six-class foreground validation IoU epoch
at **672 × 384**, then evaluates `best.pt` on the original grid. Case-equal
scoring aggregates within each video before averaging each class's defined
video scores. Absent truth with false positives counts as zero; absence in both
is excluded. Truth-positive case counts are fixed, but prediction-dependent
scored-case counts can differ: for example, plate scores 10 cases in 001 versus
9 locally because of an additional false-positive-only case, with 6
truth-positive cases in both.

| Runtime component | Local reference | All cloud runs |
| --- | --- | --- |
| Python | 3.14.2 | 3.11.13 |
| Torch | 2.14.0 | 2.8.0+cu126 |
| Torchvision | 0.29.0 | 0.23.0+cu126 |
| NumPy | 2.5.3 | 2.3.2 |
| Pillow | 12.3.0 | 11.0.0 |

Different hardware and libraries prevent attributing local/cloud prediction
differences solely to the accelerator. Canonical manifest hashes differ because
they include absolute paths: local
`6c9197f8f17642514c282bf67d0764876d23a23cc5e1e5ad7593dd0e20b3db7b`;
cloud `884667ecacfcec7c8ba875c82aa5f78adddfa5ceb4245a8a4cd46d6ffa26ac55`.
Root normalization explains the complete manifest difference.

## One-image local inference timing

The [timing artifact](outputs/cloud-handoff/inference-benchmark.json) measures
the segmentation adapter on the **Apple M4 / 16 GB / MPS**, using one fixed
854 × 480 training image, `100_27925`, at 672 × 384 model input. Per model,
three warmups precede ten timed calls, with MPS synchronization around each call.

| Collected model | Median adapter latency | 90th percentile |
| --- | ---: | ---: |
| 002: MobileNetV3 | 53.39 ms | 54.25 ms |
| 005: ResNet50 | 486.03 ms | 738.38 ms |

Timing includes image decode, preprocessing, device transfers, model forward,
original-grid logits/softmax, polygon extraction/filtering, and contract
validation. It excludes model loading, file export, camera capture, video
decoding, network, browser rendering, and display. The same image was repeated
with warmed caches and model order was sequential; this is not an accuracy
measurement or a general inference benchmark. It establishes **neither A100,
phone, live-camera, nor AR-glasses throughput**. Reciprocals of these latencies
must not be reported as measured streaming frame rates.

## Selection and next hypotheses

Retain the local demo checkpoint. The [annotation-selected before/after sheet](outputs/cloud-002-longer/comparison-small.png)
and [sidecar](outputs/cloud-002-longer/comparison-small.json) compare it with
002 for diagnosis; their examples are selected from annotations rather than
ranked by model quality and do not supply a new selection criterion.

The next hypotheses are native-resolution inputs, a separately tested objective
focused on small structures, and additional reviewed pixel annotations. These
require a future experiment or data effort; none is implemented or run in this
five-job set. More ground-truth data is constrained by the public release,
as described in the [next-data assessment](CLOUD_EXPERIMENTS.md#next-data-phase--outside-these-five-runs).

One candidate objective is [generalized Dice loss](https://arxiv.org/abs/1707.03237),
which was studied for highly imbalanced segmentation. That research motivates
a hypothesis to test here, not an established improvement for Endoscapes.
Keep native 854 × 480 inputs and a loss change in separate comparisons against
the same cloud control; do not silently combine them or retune on test cases.

## Artifact and job trace

All jobs ran in project `eastwest72hack26bos-501`, region `us-central1`,
and reached `JOB_STATE_SUCCEEDED`. The completion times below come from each
collected runner completion record, rather than a claimed exact Vertex end time.

| Run | Vertex custom job ID | Runner completed (September 13, UTC) |
| --- | --- | --- |
| 001 | `3017512501681061888` | 03:42:56 |
| 002 | `6848949884666511360` | 03:52:52 |
| 003 | `2909989060577591296` | 03:59:44 |
| 004 | `6893422930986795008` | 04:08:41 |
| 005 | `7671982716568469504` | 03:56:59 |

| Shared submitted bundle | SHA-256 |
| --- | --- |
| Source | `2fbf83b52c1a836f3155adac76e063c13f1d1092983f8bc4513233461955a86e` |
| Data | `831c81a7d6a8bfb9e9038ce892584f7976dc9e6451026ca60ba11b66a08f56e7` |

| Selected checkpoint | SHA-256 |
| --- | --- |
| 001 | `55d2571b3d45e98a2c60c28ac3b8db58d2c72f05741ecedf9918776c1aef34a3` |
| 002 | `929089e06153e88c3859c84e8b4a1939676ceca49066409f8295b198376148b2` |
| 003 | `d47c5250738bd1d4f4393f1be30d4345cda4a1a17e88ac52c5cdc56a2aec0118` |
| 004 | `fed2dc99d305351757b9a34b9866a97cb4552b242ef41bfef51250d0c46d3b2a` |
| 005 | `25605968e7d9d81cdcf601baee3738c43f293dfbb44c6bd118adcefed88fa25f` |

The [launch manifest](outputs/cloud-handoff/launch-manifest.json) records immutable
bundle URIs, hashes, inventories, and filled job configurations. Each collected
run contains `cloud-completion.json`, `cloud-collection.json`, configuration,
history, logs, runtime information, selected model version, and attempt identity.
Completion records for [001](outputs/cloud-001-control/cloud-completion.json),
[002](outputs/cloud-002-longer/cloud-completion.json),
[003](outputs/cloud-003-augmentation/cloud-completion.json),
[004](outputs/cloud-004-case-balanced/cloud-completion.json), and
[005](outputs/cloud-005-resnet50/cloud-completion.json) link exact artifact
hashes and GCS URIs. Metadata retains cloud paths; the current checkout is not
a substitute for the submitted source bundle. This audit performed no training,
test evaluation, or additional dataset acquisition.
