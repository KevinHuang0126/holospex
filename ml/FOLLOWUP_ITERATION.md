# September 14 accuracy follow-ups — complete

Historical experiment-window report. The winner was subsequently promoted at the
user's request; [CURRENT_MODEL.md](CURRENT_MODEL.md) defines the current selection.
Statements below about leaving the demo unchanged describe window closure.

Subsequent matched regularization tests are complete; see [the nine-cell comparison](REGULARIZATION_TESTS.md). Its three paired seeds supersede this window's two-pair mean estimates.

All **four approved Spot A100 jobs succeeded**, completing **212/212 epochs**.
The controller closed at **17:59:46 UTC**, before its fixed **18:34:19 UTC**
deadline. All owned jobs are terminal, and the controller and its temporary
keep-awake helper have exited. No cancellations, failures or partial schedules
were needed. All **72 completion artifacts** passed independent size/SHA checks.

The best research checkpoint improves native foreground IoU from **52.8251%
to 53.3060% (+0.4809 percentage points)**. It changes AdamW weight decay from
0.01 to 0.05 while preserving the 407-image data, source, initialization and
53-epoch training recipe. Stronger Lovasz (0.5) did not improve the primary metric.

| Recipe / seed | Selected / completed | Native foreground IoU | Small pooled | Small equal-case |
| --- | --- | ---: | ---: | ---: |
| WD 0.05 / 42 | 43 / 53 | 53.3060% | 36.4221% | 32.8865% |
| Lovasz 0.5 / 42 | 32 / 53 | 52.4804% | 35.3664% | 33.0491% |
| WD 0.05 / 43 | 38 / 53 | 52.8260% | 35.6034% | 31.9369% |
| WD 0.05 / 44 | 40 / 53 | 51.4790% | 33.7408% | 31.2816% |

Both matched seeds improve: seed 42 gains **+0.4809** foreground points and
seed 43 gains **+0.9088**. Their mean gains are **+0.6949 foreground**, **+0.9733
small pooled**, and **+0.5633 small equal-case** points. The three-seed weight-decay
recipe averages **52.5370% ± 0.9472 points sample SD**, with a **51.4790–53.3060%**
range. Seed 44 has no same-data baseline; do not compare the unequal two- and
three-seed averages as a treatment effect.

The best checkpoint improves artery and triangle IoU by 2.0770 and 3.1528 points,
but plate IoU falls 1.4733 points and its recall falls 7.5970 points. Small-case
IoU improves in five of ten cases. Across the two matched seeds, mean triangle
IoU gains 3.0239 points, artery gains 1.1399, and plate loses 0.2963 points.
This supports a modest repeatable gain with class/case tradeoffs. The **75%
target was not reached**. No test inference or clinical validation was performed.

## Verified artifacts and usage

- [Final report](outputs/followups-20260914-1509/reports/final/RESULTS.md),
  [comparison chart](outputs/followups-20260914-1509/reports/final/comparison.png),
  [learning curves](outputs/followups-20260914-1509/reports/final/learning-curves.png),
  and [class/case chart](outputs/followups-20260914-1509/reports/final/class-case-tradeoffs.png).
- [Independent aggregate audit](outputs/followups-20260914-1509/independent-result-audits/window-summary.json)
  binds all four complete schedules, exact source/configuration/checkpoint identities,
  frozen native validation counts and live TRAIN/VAL image/mask fingerprints.
- [Closure verification](outputs/followups-20260914-1509/closure-verification.json)
  and [final cloud states](outputs/followups-20260914-1509/cloud-final-jobs.json)
  establish terminal compute. Reporting and the final local loading check occurred
  after controller closure; they did not extend cloud training or launch more jobs.
- [Best research checkpoint](outputs/followups-20260914-1509/results/followup-20260914-1509-001-moco-wd005/train/best.pt):
  SHA-256 `9dc50d58fb2f605f5fc2a00652d7dae662f7584ab08e37502fb179b152873c1b`, selected epoch 43/53.
  Strict CPU model loading passed. To use it, pass the following checkpoint path:

```text
--checkpoint ml/outputs/followups-20260914-1509/results/followup-20260914-1509-001-moco-wd005/train/best.pt
```

The selected demo at `ml/weights/current/best.pt` and existing video predictions
retain their previous identities; research output was not automatically promoted.
All 44 controller tests and the full 342-test workspace ML suite passed before
launch. Worker training code was unchanged, and the final charts were visually
checked. Unrelated inference/latency edits were preserved.

## Next evidence-backed comparison

In a newly authorized window, first complete the missing same-data **seed-44
control at weight decay 0.01**, holding the 407 images, 53 epochs and every other
recipe setting fixed. This separates seed difficulty from the regularization
effect; the historical 361-image/60-epoch seed 44 is not a matched control.
Then test intermediate weight decay **0.025 versus 0.05** across matched seeds
to see whether plate/duct recall can recover while retaining triangle precision.
This is a hypothesis, not an additional launch in the completed budget.

## Reproduce the report

Use a new output directory; the report helper refuses overwriting prior reports:

```sh
.venv/bin/python ml/outputs/followups-20260914-1509/report-tools/followup_report.py \
  --output-dir ml/outputs/followups-20260914-1509/reports/new-report
```

The saved controller state is complete and must not be restarted. Historical
operator notes below describe the approved plan, not active or pending work.

## Historical approved plan

The approved cap is two hours from the recorded start, four A100 Spot launches,
and at most two simultaneous workers. Reserve the final two launches for seed
repeats after both initial results are audited. The user approved this concrete
cap with "yes". The absolute UTC deadline is fixed in the new controller state.

| Initial candidate | Only change from batch002 seed 42 | Hypothesis |
| --- | --- | --- |
| Stronger weight decay | AdamW decay 0.01 → 0.05 | Reduce the measured late overfitting. |
| Stronger Lovasz term | Main-head Lovasz weight 0.25 → 0.5 | Test whether the successful IoU objective has further benefit. |

Both use fresh training from the same surgical MoCo ResNet50 backbone, with
53 epochs, 672 × 384 input, batch size 2, LR 0.0003, backbone multiplier 0.1,
three-epoch warmup and cosine decay, balanced CE, auxiliary weight 0.4,
uniform sampling and no augmentation. These are fresh training runs, not
continuations of the current checkpoint. All 407 training images and approved
partial targets remain unchanged. Unknown and conflicting pixels stay ignored.

The reference seed-42 training loss decreases from 0.06467 at selected epoch 34
to 0.04340 at epoch 53, while validation CE increases from 0.27535 to 0.43896.
The matched historical MoCo CE → CE plus 0.25 Lovasz trial improved foreground
IoU from 51.0913% to 52.0409%. Neither stronger weight decay nor the 0.5 Lovasz
coefficient has yet been evaluated in this recipe.

Triangle false positives are a decision criterion alongside the primary metric:
the current winner has triangle precision 35.4133%, recall 81.6015% and IoU
32.7948%. Most of the increase in triangle false positives comes from background
pixels. Stronger Lovasz could worsen this tradeoff; improvement is a hypothesis.
Larger inputs and mild augmentation already failed matched MoCo comparisons,
so they are lower priorities than these new controlled changes.

Existing histories indicate approximately 23 minutes of training per 53-epoch
run. Budget 28–32 minutes through worker completion plus collection contingency.
After both initial candidates finish, freeze the highest independently verified
full recipe and repeat it at seeds 43 and 44. Preserve the full schedule and
report any duration-limited run separately. Compare seed 43 to the existing
batch002 seed-43 control; seed 44 has no same-data historical control.

## Frozen evaluation and operation

The primary metric remains six-class foreground macro IoU from pooled confusion
counts on all 75 validation images from ten cases at the original 854 × 480
annotation grid. Exclude background only from the class average, retain its
false positives, and ignore source 255. Secondary measures are per-class IoU,
precision/recall, pooled small-anatomy IoU and equal-case small-anatomy IoU.
The existing worker selects checkpoints on its input-grid validation metric;
the selected checkpoint's independently audited native-grid result determines
the comparison between runs. The test split is excluded from search.

The historical 75% target remains an early-stop threshold, not an expected
outcome. Repeated validation selection and a small seed group do not establish
significance or clinical validity. Research candidates do not automatically
replace `ml/weights/current/best.pt` or existing video predictions.

The plan reuses already uploaded immutable source, base-data and reviewed-data
bundles; local hashes and cloud object metadata match. This preserves a matched
worker implementation while separate inference/latency edits are in progress.
The local controller will use `ml/cloud/autonomous_loop.py`, its persisted
submission identities, bounded collection and owned-job cancellation. It must
wait for all initial candidate results before freezing the repeat group.

Preparation artifacts are under
[`outputs/followups-20260914-1509/`](outputs/followups-20260914-1509/), including
the non-runnable [launch plan](outputs/followups-20260914-1509/launch-plan.json)
and [bundle checks](outputs/followups-20260914-1509/preflight/bundle-verification.json).
The [independent baseline audit](outputs/followups-20260914-1509/preflight/AUDIT.md)
records all 36 verified completion artifacts, live train/validation file hashes,
and confusion metrics recomputed independently from the saved counts. It also
checks validation support against all 75 original masks: 30,683,409 scored and
60,591 ignored pixels. Its script and full JSON receipt are preserved beside it.
The controller now enforces the comparison barrier before freezing a repeat
group, including direct-launch attempts; already frozen repeats can run together.
All **44 controller tests** and **342 full-workspace ML tests** pass. The full
[test log](outputs/followups-20260914-1509/preflight/ml-tests.log) includes the
concurrent inference/latency work present in the shared workspace. Cloud workers
will continue to use the unchanged historical source bundle.
The approved start created [state.json](outputs/followups-20260914-1509/state.json)
there with its absolute deadline, limits and authorization. Closure requires terminal owned
jobs, stopped controller, verified result report, comparison chart and updates
to this guide, `STATUS.md` and `EXPERIMENT_LOG.md`.
