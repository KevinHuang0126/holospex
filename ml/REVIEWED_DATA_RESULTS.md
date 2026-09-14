# Reviewed-data training results

The first reviewed pilot supports **another bounded labeling batch**. All three paired seeds improved both small-anatomy pooled IoU and equal-case IoU with nearly equal optimizer-update budgets. The cystic plate remains a weakness: its pooled result barely improved, two seeds regressed, and its mean equal-case IoU and precision fell.

All six Vertex A100 jobs succeeded. The 93 published artifacts passed SHA-256 and size verification. The comparison uses the same 75 validation images from ten surgical cases; no test evaluation or demo-video scoring was performed.

## Aggregate result

Means across seeds 42, 43 and 44. IoU measures predicted/annotated region overlap; higher is better. Pooled metrics aggregate pixels before scoring. Equal-case metrics give each surgical case equal weight for each defined class score.

| Original-grid metric | Original data | With reviewed labels | Change |
| --- | ---: | ---: | ---: |
| Small-anatomy pooled IoU | 24.40% | 27.23% | +2.83 pp |
| Small-anatomy equal-case IoU | 22.59% | 24.65% | +2.06 pp |
| Six-class foreground IoU | 43.26% | 45.29% | +2.03 pp |
| Six-class foreground Dice | 55.33% | 57.83% | +2.50 pp |

All three pairs improved both small-anatomy measures. Pooled gains ranged from **+2.18 to +3.88 percentage points**; equal-case gains ranged from **+0.62 to +3.71 points**. This meets the criterion frozen before training: at least one point of mean improvement in both measures and positive changes on both in at least two seed pairs. It is a practical pilot criterion, not a significance test.

## Per-class outcome

| Small anatomy | Pooled IoU: original → reviewed | Precision: original → reviewed | Recall: original → reviewed |
| --- | ---: | ---: | ---: |
| Cystic duct | 37.95% → 42.11% | 66.60% → 62.43% | 46.88% → 56.69% |
| Cystic artery | 12.73% → 17.67% | 28.32% → 31.03% | 18.72% → 29.59% |
| Cystic plate | 18.94% → 19.28% | 37.24% → 32.59% | 27.89% → 32.44% |
| Triangle dissection | 27.98% → 29.85% | 37.99% → 38.98% | 51.72% → 57.32% |

- **Artery is the clearest gain:** mean IoU rises 4.93 points, recall 10.87 points and precision 2.71 points. Duct also gains 4.16 IoU points, with higher recall but lower precision.
- **Plate improvement is not reliable:** pooled IoU changes only +0.33 points; it falls in seeds 43 and 44. Equal-case plate IoU falls **18.29% → 17.19%** and equal-case precision **36.56% → 28.55%**. Plate boundaries and confusion with neighboring structures should be a specific review focus.
- Triangle has a positive mean result, but its precision/recall tradeoff varies between seeds. In seed 42 alone, recall rises 20.12 points while precision falls 6.40 points. Do not infer uniform behavior from the mean.

## Every paired seed

| Seed | Original selected epoch | Reviewed selected epoch | Small pooled IoU: original → reviewed | Small equal-case IoU: original → reviewed |
| --- | ---: | ---: | ---: | ---: |
| 42 | 34 | 33 | 23.78% → 27.65% | 22.72% → 26.43% |
| 43 | 26 | 38 | 24.32% → 26.50% | 23.00% → 23.62% |
| 44 | 23 | 38 | 25.11% → 27.53% | 22.05% → 23.91% |

## What was controlled

The [frozen plan](REVIEWED_DATA_ITERATION.md) compares 343 original train images from 30 cases with 361 images from 48 cases. The added 18 images contain 49 anatomy-reviewed accepted/corrected masks. Original samples and holdouts are byte-identical across arms. Unknown surroundings and conflicting overlaps remain ignored, not background.

Both arms use DeepLabV3–MobileNetV3-Large at 672 × 384, fresh generic pretrained initialization, balanced cross-entropy, batch size 2, learning rate 0.0003, frozen BatchNorm, no augmentation and uniform sampling. Original-data runs use 42 epochs (7,224 updates); reviewed-data runs use 40 (7,240 updates), a **0.22% update-budget difference**. The original controls have two more validation checkpoint-selection opportunities. The same class-weight formula is recalculated from each arm’s training masks; adding data also changes shuffle order.

The added masks provide 203,344 scored pixels at training resolution, raising the four small-class pixel counts by about 5–14%. Training epochs averaged 15.95–17.82 seconds on A100 workers. Jobs ran in two waves because the project has a 42-core Spot training quota and each A100 worker requires 12 cores.

Every selected checkpoint uses the existing maximum resized-grid six-class foreground-IoU rule. Final comparisons restore logits to native annotation dimensions before argmax and apply the same ignored-pixel policy. The report verifies actual per-epoch updates/pixels, checkpoint identity, runtime settings and input fingerprints; it independently recomputes scores from confusion matrices.

## Visual check

The [seed-42 comparison sheet](outputs/reviewed-handoff/seed42-comparison-small.png) and [sidecar](outputs/reviewed-handoff/seed42-comparison-small.json) show the same four median-annotation-area examples used by the established comparison rule. Selection does not inspect model quality. Both models infer on the whole frame; only the display is cropped, and HUD confidence/contour filters are bypassed.

The four whole-frame focus IoUs are duct **0.587 → 0.477**, artery **0.044 → 0.312**, plate **0.000 → 0.000**, and triangle **0.061 → 0.192**. These examples preserve regressions and persistent misses alongside gains; they are not the aggregate result.

## Recommendation and limits

Review another **40–60 images across 20–30 new training cases**, then repeat the same controlled comparison before committing to a much larger queue. Prioritize informative artery and duct examples while giving plate boundaries and neighboring-structure confusion extra attention. Keep difficult or uncertain regions explicitly uncertain; do not approve them simply to increase the sample count. This batch size is a practical next step, not an estimated optimum or a promised gain.

This pilot supports the value of adding these reviewed labels under the tested recipe. It does not isolate human correction from the value of additional images, because there is no matched unreviewed-proposal arm here. Three seeds and ten repeatedly used validation cases do not establish statistical significance or broad generalization; the original test set was inspected in earlier work and was not evaluated again here.

Keep `small-004-resolution/best.pt` as the selected demo checkpoint for this task. Preserve all three reviewed-data checkpoints as candidates; do not silently select the luckiest seed or replace the teammate’s demo assets. This experiment establishes a labeling decision, not reviewed lesson answers or clinical validation.

## Artifacts and reproduction

- [Machine-readable comparison](outputs/reviewed-handoff/comparison.json)
- [Launch manifest](outputs/reviewed-handoff/launch-manifest.json), including immutable bundle hashes and exact configurations
- [Launch audit](outputs/reviewed-handoff/launch-audit.json)
- [Final independent results audit](outputs/reviewed-handoff/final-independent-audit.json)
- [Final Vertex job states](outputs/reviewed-handoff/final-job-status.json)
- [Prepared-data policy and provenance](PARTIAL_TRAINING.md)

The frozen training source passed 192 tests after extraction; the final workspace passes 205 ML tests, including the post-training summary and data-identity regressions. Original preparation artifacts retain their historical `trainingStarted: false` fields; these separate run records establish the subsequent completed training.

```sh
.venv/bin/python ml/cloud/summarize_reviewed_results.py \
  --results-root ml/outputs/reviewed-handoff/results \
  --launch-manifest ml/outputs/reviewed-handoff/launch-manifest.json \
  --output ml/outputs/reviewed-handoff/comparison-rerun.json
```

| Run | Vertex job ID | Selected checkpoint SHA-256 |
| --- | --- | --- |
| `reviewed-001-base-s42` | `1003959265548828672` | `a87d74f38b933cedc4dbbcf8495b0a6b01846b059af408cc583ea1b90ba67cbd` |
| `reviewed-002-added-s42` | `6619948000879837184` | `b59cbc7d3e7fd25b74631f1e22611e92bbdf72a87867708a994b50ac8cd94e4c` |
| `reviewed-003-base-s43` | `8155675473813176320` | `603daf75d413fa770699bc1192ed4199a461e05cdb04c6d4588b3094208caf3d` |
| `reviewed-004-added-s43` | `3080963118696759296` | `43e5994751aa3f7de6dda660d41106be972cfd3522554739c37ac1980dbb404c` |
| `reviewed-005-base-s44` | `661685693868670976` | `11fa14bb05e6b4f6fe93bf25020d421160889d967cd950ac1188a344a0543988` |
| `reviewed-006-added-s44` | `8665426656636174336` | `8e4007b585ada756fa9442cb519ea462c878ddbbe934b661819e58c67f80f9ce` |

Comparison SHA-256: `f319f443a77cf1f947d917628a4fb6fd6dfb1e2adf175e05dd89b55bc6b9d55b`.
