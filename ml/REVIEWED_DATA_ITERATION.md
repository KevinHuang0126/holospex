# Reviewed-data training comparison

Frozen before submitting jobs on September 13, 2026. The question is whether
the first anatomy-reviewed pilot improves held-out small-structure segmentation
enough to justify another labeling batch.

## Paired experiment

Train six fresh runs on Vertex A100 workers from the same pinned container and
source archive. Pair seeds 42, 43 and 44, with generic pretrained initialization.

| Arm | Training images | Training cases | Epochs | Optimizer updates |
| --- | ---: | ---: | ---: | ---: |
| Original Seg50 | 343 | 30 | 42 | 7,224 |
| Seg50 + reviewed pilot | 361 | 48 | 40 | 7,240 |

The different epoch counts intentionally match optimization budgets within
0.22%. Each epoch visits every training image once in a uniform random order.
The original-data control has two additional validation selection opportunities.
Keep its best checkpoint over the full 42 epochs; do not retrospectively choose
a different budget after looking at results.

Fixed settings: DeepLabV3 with MobileNetV3-Large; 672 × 384 input; balanced
cross-entropy; batch size 2; AdamW with learning rate 0.0003; frozen BatchNorm;
no augmentation; uniform sampling; pretrained COCO/VOC initialization. Class
weights are recomputed from each arm's exact resized training masks with the
existing formula. Report those changes as part of the expanded-data recipe.
No architecture, Dice-loss, resolution, oversampling or threshold search is
included in this experiment.

The added data contains 49 accepted/corrected masks on 18 new official TRAIN
cases, bound to the exact returned review and lead's resolution record. Use only
uncontested reviewed foreground. Unknown areas and inter-class overlaps are
ignored, never background. At 672 × 384 this contributes 203,344 labeled pixels:
89,547 duct, 32,738 artery, 38,829 plate and 42,230 triangle. These increase the
respective class pixel counts by about 8.21%, 6.50%, 5.32% and 13.61%.
The additional masks comprise about 0.231% of all scored original training
pixels; 18 more cases alone should not be interpreted as 60% more supervision.

## Evaluation and decision

Both arms use the exact same 75 validation images from ten cases. Test data
remains unchanged and is not evaluated in this iteration. The existing test
split was inspected previously and is not a fresh blind holdout. No public demo
frames, training-image scores or proposal-acceptance rates select a model.

Keep the existing checkpoint selection rule: best six-class foreground macro-IoU
at the input grid. Evaluate the selected checkpoint against the original-grid
validation masks. The primary question uses both four-class small-anatomy
pooled IoU and equal-case small-anatomy IoU. Report each seed, paired differences,
mean and range; per-class IoU, Dice, precision and recall; six-class foreground
IoU; runtime; and exact source/data/checkpoint identities.

A useful positive pilot signal is an average gain of at least one percentage
point in both small-anatomy measures, with positive differences on both measures
in at least two of the three paired seeds. Check class-specific regressions and
precision/recall tradeoffs before recommending a larger batch. This is a
practical pilot criterion, not a significance test or clinical threshold.
A mixed or flat result means this small partial-label recipe is inconclusive;
a consistent regression means revise annotation coverage or training mixture
before scaling it. Neither result proves that additional reviewed labels in
general have no value.

Do not promote a single lucky checkpoint automatically. Keep the selected demo
checkpoint unchanged pending the complete aggregate evidence and visual review.
Historical 40-epoch and unreviewed-SAM trials are context, not the paired control.

## Reproduction and provenance

The ignored [launch manifest](outputs/reviewed-handoff/launch-manifest.json)
records all six filled job configurations, bundles, hashes, budget decisions and
Vertex identities. The immutable base archive is reused. A separate prepared
archive carries the exact reviewed masks, review/lead-resolution snapshots,
licenses and hash bindings for all 984 original image/mask files. Cloud staging
checks those bindings before changing only operational filesystem paths.

Training runs, checkpoints and original-grid validation artifacts will be
collected under `ml/outputs/reviewed-handoff/results/` with checksum verification.
Original review/preparation artifacts remain unchanged and retain their
historical preparation-time `trainingStarted: false` fields. Run records, not
rewritten source evidence, establish subsequent training status.
