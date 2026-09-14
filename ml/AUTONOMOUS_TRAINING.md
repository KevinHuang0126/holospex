# Four-hour autonomous training experiment

This records the completed September 13 search. The later
[reviewed batch 002](REVIEW_BATCH_002.md) produced the promoted
[current model](CURRENT_MODEL.md) at **52.8251%** native foreground IoU.

Started September 13, 2026 at 19:44:23 UTC and completed at the fixed
**23:44:23 UTC (7:44:23 PM America/New_York)** deadline. All **15 owned Vertex
jobs succeeded**, with **900 full epochs** and 15 independently audited results;
the controller is stopped and no owned jobs remain active.

The best model, run012 MoCo-initialized DeepLabV3–ResNet50 with main-head Lovasz,
reached **52.0409% six-class foreground mean IoU**, compared with **45.8717%**
for the reviewed reference (**+6.1692 percentage points**). The **75% target
was not reached**. The frozen recipe's three-seed mean was **51.5111%**, with
a 50.8585–52.0409% range. No test evaluation or automatic demo promotion occurred.
See the [final report](outputs/autonomous-20260913/reports/final/RESULTS.md),
[machine-readable summary](outputs/autonomous-20260913/reports/final/summary.json),
and [replication report](outputs/autonomous-20260913/reports/replication-results.md).
The experiment design and historical execution instructions are preserved below.

The primary metric is computed from pooled confusion counts over all 75
validation frames from 10 cases, at the original 854 × 480 annotation grid.
The background class is excluded from the macro average; background pixels
still contribute false positives when predicted as anatomy. Source-255 pixels
are ignored. All six foreground classes must have defined scores.
The 74-frame test split is not used for fitting, checkpoint selection or search.
It was evaluated for an earlier baseline, so it is not a fresh blind holdout.

## Evidence motivating this search

The preceding [experiment log](EXPERIMENT_LOG.md) records 25 completed runs.
The strongest native-grid foreground result was about 46%, and the strongest
reviewed-data run reached 45.87%. Longer constant-rate runs increasingly fit
training labels while their validation loss worsens. The tested generalized
Dice, case sampling and ResNet50 changes did not improve the aggregate result.

The new search tests learning-rate decay, conservative fine-tuning, and a
stride-four decoder that retains low-level image features. Learning-rate
scheduling is motivated by the [Torchvision segmentation training recipe](https://github.com/pytorch/vision/blob/main/references/segmentation/train.py).
The decoder is a local MobileNet implementation inspired by
[DeepLabV3+](https://arxiv.org/abs/1802.02611), not a published anatomical model.
These are hypotheses, not assumed improvements. The new decoder uses GroupNorm
for fresh layers while preserving frozen pretrained BatchNorm.

## Frozen data and first candidates

All candidates use the already prepared 361-frame training manifest, including
49 human-approved partial masks across 18 added images. Existing validation and
test identities are preserved. Unknown pixels remain ignored; proposals from
the separate ongoing annotation job do not enter this search automatically.

| Candidate | Initialization | Input | Epoch budget | Learning rate |
| --- | --- | --- | ---: | --- |
| Gentle refinement | Reviewed seed-42 best checkpoint | 672 × 384 | 40 | 3e-5, cosine decay |
| Larger scheduled input | Generic pretrained DeepLabV3 | 896 × 512 | 80 | 3e-4, cosine decay |
| Detail decoder | Generic pretrained encoder/context, fresh decoder | 672 × 384 | 80 | Head 3e-4, backbone 3e-5, 3-epoch warmup and cosine decay |
| Lower initial rate | Generic pretrained DeepLabV3 | 672 × 384 | 80 | 1e-4, cosine decay |

All retain balanced cross-entropy, AdamW weight decay 0.01, batch size 2,
uniform frame sampling and initially no augmentation. Cosine decays toward zero
over optimizer updates. Warm-starting resets optimizer state, records the exact
source checkpoint hash, and verifies matching training/validation case sets;
it does not claim exact training resumption.

After these trials, verified native-grid results determine the leader. The
controller can refine its weights, test the detail decoder at 896 × 512,
test mild augmentation or stronger weight decay, test exact-native geometry,
and replicate the leading settings with new seeds. Every choice records its
reason and concrete configuration before submission. The search is exploratory
and repeatedly selects on validation; it does not establish clinical validity.

## Execution and stopping

The controller is [cloud/autonomous_loop.py](cloud/autonomous_loop.py). Its
ignored state, immutable bundles, jobs, event log, collected results and status
are under `ml/outputs/autonomous-20260913/`. It uses the existing private GCS
bucket and the existing A100 SPOT Vertex configuration, with at most two owned
jobs running simultaneously. It never cancels unrelated project jobs.

Each submission has a persisted intent, unique name and fixed source/data
hashes. Uncertain submissions are reconciled by exact name instead of repeated.
Checkpoints and coherent epoch records are uploaded during training. Completed
artifacts are collected with byte-count/SHA-256 checks; an independent audit
recomputes validation metrics before the target can count as reached.

No new trial starts with less than eight minutes remaining. A per-run duration
budget preserves a completed epoch before evaluation; late trials receive an
explicitly shorter epoch budget. At the four-hour deadline or a verified target,
the controller requests cancellation of its remaining jobs and writes final
status. Interrupted and duration-limited trials remain labeled as such.

Result discovery and downloads share a bounded time budget, reduced to leave
30 seconds before the absolute deadline. A timed-out collection removes its
partial staging directory and can be retried with the same hash checks. After
all owned jobs are terminal, a separate two-minute collection pass audits late
successes; any remaining collection failure is recorded for manual recovery.
The final local supervisor and training changes passed the full 312-test ML suite. Worker
source versions remain recorded individually; restarting the supervisor does
not replace the immutable source of already submitted jobs.

The final seed pair uses a validated `replicate_best_full` queue marker. At
the first repeat's launch, the controller selects the highest verified native
IoU among owned runs that completed their full requested recipe, then freezes
that recipe and its source/checkpoint identity for the named group. Both
repeats change only seed, label and recorded rationale. A later leader cannot
silently change the second member's recipe. Markers allow no training, data
or budget overrides. Full schedules are preserved under the existing runtime
cap; a repeat counts as complete only when every requested epoch finishes.
This addition passed the full **312-test ML suite** and changes only the local
supervisor, leaving worker source version 4 intact.

No demo checkpoint is promoted automatically. Results are recorded in the
experiment log with per-class and per-case tradeoffs and checkpoint identities.
The 75% goal is ambitious relative to the measured baseline and may remain unmet.

## Prepared follow-up options

The optional `ce_lovasz` objective adds 0.25 times foreground Lovász-Softmax to
the main head while retaining balanced CE on the main and auxiliary heads.
It pools scored pixels across each batch, averages foreground classes present
in the targets, excludes source 255 before sorting, and leaves absent-class
false-positive penalties to CE. The formulation and MIT attribution follow
the [authors' implementation](https://github.com/bermanmaxim/LovaszSoftmax).
This is distinct from the previous inverse-volume generalized Dice trial.

A safely loaded [SelfSupSurg DINO ResNet50 checkpoint](https://github.com/CAMMA-public/SelfSupSurg)
is also available as an optional backbone initialization. The official source
checkpoint is licensed CC BY-NC-SA 4.0; all source and derived files are ignored
by Git. Its documented pretraining uses Cholec80 training cases 1–40. The
[CAMMA dataset crosswalk](https://github.com/CAMMA-public/camma_dataset_overlaps)
was checked against all actual Seg50 validation and test cases, with empty
intersections. Its 318 backbone tensors exactly match the current ResNet50
backbone. The generic COCO context and auxiliary head initialization are retained.
This is preparation evidence, not an accuracy result.

The same adapter also supports the official surgical MoCo-v2 ResNet50
checkpoint, using the pinned repository's
[h001 pretraining configuration](https://github.com/CAMMA-public/SelfSupSurg/blob/8a03d28948e59471cc8ea865d56632eda048079c/configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h001.yaml).
Its Cholec80 training-case provenance and held-out case separation were checked
independently. A matched 60-epoch trial changes only the surgical pretraining
method relative to the DINO arm. The authors' older Endoscapes F1 results use a
different class set and metric, so they are not numerical baselines for this
search. Derived weights and their source checksums remain Git-ignored.

The first completed warm-start trial selected epoch 1 and essentially tied its
initialization. Its auxiliary CE contributes most of the late scalar training
objective, which motivates an optional auxiliary-weight ablation. This is not
evidence that its gradients are harmful. A main-only trial will change the
auxiliary weight from 0.4 to zero while matching the original checkpoint,
40-epoch cosine trajectory, grid, seed, and learning rate of the first run.

The initial detail-decoder trial uses a backbone rate one tenth of the main
rate. A matched 672 × 384 trial with backbone multiplier 1.0 is therefore
planned before increasing decoder resolution. A standard MobileNet control
will use the same 80-epoch schedule and three-epoch warmup to distinguish the
decoder change from optimization. Exact-native 854 × 480 input remains a
separate geometry experiment. Actual launches and changed priorities are
recorded in the controller state and event log.

Source snapshots are immutable. Version 1 passed 243 extracted-source ML tests;
version 2 added Lovász/deadline support and passed 267; version 3 added surgical
backbone/reference support and passed 283; version 4 added the auxiliary-loss
ablation and passed 293. The controller records each job's
actual source identity. Future snapshots can be reproduced with:

```sh
.venv/bin/python ml/cloud/package_autonomous_source.py \
  --output-dir ml/outputs/autonomous-20260913/source-N
```

The package utility includes the review HTML fixture, rejects runtime/private
artifacts, verifies extracted file hashes, and runs the full ML suite before
writing a source receipt. It does not upload or submit a job. `next-plan.json`
supports validated once-only plan changes while cloud jobs continue running.

## Local operation and reports

This session was launched and resumed after interruptions with:

```sh
.venv/bin/python ml/cloud/autonomous_loop.py \
  --state ml/outputs/autonomous-20260913/state.json
```

Resumption keeps the saved deadline and owned job identities. The controller
uses a process lock to prevent duplicate owners and refuses completed states.
`STATUS.md` beside the state file is refreshed as jobs are collected and audited.
The September 13 state is a historical record after its deadline; it must not
be reused to authorize another training window. A future authorized search
needs its own state, unique run prefix, deadline and immutable input references.

A report snapshot can be created in a fresh output directory:

```sh
.venv/bin/python ml/cloud/summarize_autonomous_results.py \
  --state ml/outputs/autonomous-20260913/state.json \
  --output-dir ml/outputs/autonomous-20260913/reports/new-snapshot
```

This rechecks collected hashes and native confusion metrics, then writes
`RESULTS.md` and `summary.json`. It includes pending and failed runs, the
reviewed reference, per-class tradeoffs, source/checkpoint identities, and
separate training and worker durations. A snapshot of an active search is
explicitly labeled with its timestamp; it does not query cloud state or run
new inference.
