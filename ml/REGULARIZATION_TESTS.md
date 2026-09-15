# Matched weight-decay tests — complete

Historical experiment-window report. The winner was subsequently promoted at the
user's request; [CURRENT_MODEL.md](CURRENT_MODEL.md) defines the current selection.
Statements below about leaving the demo unchanged describe window closure.

All four approved Spot A100 jobs succeeded: **212/212 epochs, 72 independently
verified artifacts, no failed or partial runs**. Training ended by **20:50:41 UTC
on September 14**. A stalled local checkpoint transfer was recovered, and the
controller closed at **21:35:55 UTC**, before the fixed **21:45:36 UTC** deadline.
Every owned cloud job is terminal; controller, keep-awake helpers and orphaned
download workers are confirmed absent. The new tests did not beat the existing
**53.3060% native foreground IoU** research checkpoint.

## Complete matched comparison

Four new runs fill the missing 0.01 seed-44 control and all three intermediate
0.025 cells. Five previously completed controls were independently re-audited.
Scores below use six-class foreground macro IoU at native resolution.

| Weight decay | Seed 42 | Seed 43 | Seed 44 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| 0.01 | 52.8251% | 51.9172% | 51.6575% | 52.1333% ± 0.6130 pp |
| 0.025 | 51.9910% | 52.2905% | 51.4658% | 51.9158% ± 0.4175 pp |
| 0.05 | 53.3060% | 52.8260% | 51.4790% | 52.5370% ± 0.9472 pp |

The 0.025 recipe loses foreground IoU to 0.05 in all three matched seeds.
Mean changes versus 0.05 are **−0.6213 foreground**, −0.8647 pooled small and
−0.5991 equal-case small percentage points. The hoped-for plate recovery did
not occur: mean plate IoU falls 1.8288 points versus 0.05.

Completing the third control puts 0.05 versus 0.01 at **+0.4037 foreground**,
+0.5415 pooled small and +0.8686 equal-case small points on average. Seed 44
regresses 0.1785 foreground and 0.3221 pooled-small points. The largest mean
class gain is triangle IoU, +2.1264 points. Plate IoU is nearly flat (−0.0263),
with precision +2.2718 and recall −4.5402 points. Some validation cases regress.
These three seeds and repeated validation selection do not establish statistical
significance, unseen-video accuracy or clinical validity. The **75% target was
not reached**.

[Final report](outputs/regularization-20260914-1945/reports/final/RESULTS.md),
[comparison chart](outputs/regularization-20260914-1945/reports/final/native-score-grid.png),
[class/case tradeoffs](outputs/regularization-20260914-1945/reports/final/matched-tradeoffs.png),
[learning curves](outputs/regularization-20260914-1945/reports/final/learning-curves.png)
and [independent summary](outputs/regularization-20260914-1945/independent-result-audits/comparison-summary.json)
contain the full three-by-three results. All charts were visually checked.

## Frozen recipe and evaluation

Every cell used 53 epochs, the same 407 training images from 73 cases,
672 × 384 input, batch size 2, surgical MoCo ResNet50, balanced CE plus 0.25
main-head Lovasz, auxiliary weight 0.4, AdamW LR 0.0003, backbone multiplier
0.1, three-epoch warmup/cosine, uniform sampling and no augmentation. Only
weight decay and seed vary. All new runs started from identical backbone
weights; none resumed a prior training checkpoint.

Validation uses all 75 frames from ten fixed cases at original 854 × 480
annotation resolution. Foreground macro IoU averages six foreground classes
from pooled confusion counts, retains background false positives and ignores
source 255. Secondary metrics include per-class precision/recall, pooled and
equal-case small-anatomy IoU. The worker selects checkpoints on input-grid
validation; final comparisons independently recompute native-grid metrics.
No test inference, label changes or automatic demo promotion occurred.

The source, data and prepared-package SHA-256 identities are respectively
`5a41ed8a819ae6d569f8439549f1963a035692a8d4007bb03ae36a09d5185648`,
`831c81a7d6a8bfb9e9038ce892584f7976dc9e6451026ca60ba11b66a08f56e7` and
`935e4b68e2aed4baa1251acb11f6ed02953183f1651a5fcbebd3a975e301fe62`.
Exact configs, submitted job identities, class mapping and split fingerprints
remain in the [approved state](outputs/regularization-20260914-1945/state.json)
and independent audit files. Live TRAIN/VAL file hashes match all references.

## New runs and verified artifacts

| Run suffix | Vertex job ID | Selected / completed epochs | Checkpoint SHA-256 |
| --- | --- | --- | --- |
| 001-control-wd001-s44 | 1001962552432787456 | 25 / 53 | `135efc4cdc3decc7c65881cfa9838418288535b6c0a30fbb5f9cea5c663916c7` |
| 002-middle-wd0025-s42 | 2274229447164952576 | 34 / 53 | `324b482e7dd4b0c71036a9e93c1a12fe48d0c3bae78b3c4d5b350b54e1449dd8` |
| 003-middle-wd0025-s43 | 1101534325444182016 | 42 / 53 | `539222d1fcb35a06a9d6cdc3d1baec728990f2fb67bbb399553a26e9b10d3621` |
| 004-middle-wd0025-s44 | 4504003843922591744 | 41 / 53 | `e403594d9587cee7b35846ce7f0ce1b371624fd7a9570afdd63274e187448ba4` |

Each suffix follows `reg-20260914-1945-` under
`ml/outputs/regularization-20260914-1945/results/`; each checkpoint is
`train/best.pt`. All 18 artifacts per run, best/last checkpoint metadata,
53-epoch histories and native confusion counts passed independent checks.
All four new best checkpoints passed strict CPU model loading.

The best research checkpoint remains
[weight decay 0.05, seed 42](outputs/followups-20260914-1509/results/followup-20260914-1509-001-moco-wd005/train/best.pt),
SHA-256 `9dc50d58fb2f605f5fc2a00652d7dae662f7584ab08e37502fb179b152873c1b`.
It was loaded strictly on CPU again. The selected demo at
`ml/weights/current/best.pt` retains SHA-256
`b406ed42ab0394edba22e1dde0edc2865a6346adb61c4bea7ab0bc00d08e1911`.

The [closure receipt](outputs/regularization-20260914-1945/closure-verification.json)
records live terminal cloud states, process exit checks, model loads and report
hashes. The local ML suite passed **342 tests** before this window; no execution
code changed during this window. A local transfer timed out while captured
pipes remained open in orphaned gcloud workers. Only verified owned workers
were removed; the existing controller collected both results and exited.
The [recovery record](outputs/regularization-20260914-1945/preflight/collection-recovery.json)
preserves that operational limitation. No new cloud job or deadline extension
was needed.

## Next evidence-backed experiment

Before trying another decay value, audit native-grid checkpoint selection with
twelve predeclared retained snapshots of the unchanged 0.05 recipe:

| Seed | Input-grid best | Two alternate top epochs | Final epoch |
| --- | ---: | --- | ---: |
| 42 | 43 | 51, 47 | 53 |
| 43 | 38 | 39, 42 | 53 |
| 44 | 40 | 41, 35 | 53 |

Epochs 40 and 41 in seed 44 differ by only 0.0012 input-grid percentage points;
observed native-minus-input shifts at selected checkpoints reach 0.0345 points.
A ranking reversal is plausible, but this does not imply a large accuracy gain
or explain the plate deficit. Freeze this snapshot list and metric policy before
evaluation; report native results and plate/triangle/case tradeoffs for every
snapshot. Retain the test split outside selection.

All twelve exact coherent snapshot records exist. Six best/final checkpoints
are local with verified byte hashes; six alternate weight files remain in cloud
storage (1,010,111,778 bytes total). Exact record and object metadata checks are
in the [availability manifest](outputs/regularization-20260914-1945/preflight/selection-snapshot-availability.json).
Their weight bytes were not downloaded or evaluated during this window.
Reusing the three audited best results leaves nine evaluations × 75 frames
(675 forward passes); independently rerunning all twelve needs 900.
This proposed audit needs no new training, but was not executed as part of
these four tests. This completed window must not be restarted or extended.

## Reproduce reports and audits

From the repository root, use the saved artifacts without launching training:

```sh
.venv/bin/python ml/outputs/regularization-20260914-1945/preflight/regularization_audit.py --state ml/outputs/regularization-20260914-1945/state.json --summary
.venv/bin/python ml/outputs/regularization-20260914-1945/report-tools/regularization_report.py --state ml/outputs/regularization-20260914-1945/state.json --output-dir ml/outputs/regularization-20260914-1945/reports/reproduced
```

Use a fresh report output directory. Individual `--run RUN_NAME` audits verify
all artifact hashes, checkpoint metadata, configurations and live TRAIN/VAL
identities before rebuilding the summary. Dataset and checkpoint files remain
ignored and outside Git.

## Historical operator notes

The earlier proposal under `outputs/regularization-20260914-1930/` was blocked
before any job launched. The user then explicitly approved two hours, four
Spot A100 jobs and two concurrent workers. A fresh state fixed the window at
**19:45:36–21:45:36 UTC September 14**; prior completed/rejected states were
never reused. The interim `reports/control-complete/` snapshot is historical
and superseded by `reports/final/`. The four approved launch slots are exhausted,
and this window is closed.
