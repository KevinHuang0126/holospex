# Frozen cloud experiment plan

This plan defines five fresh training runs before inspecting their outcomes.
It separates moving the current model to CUDA, increasing the training budget,
and testing three independent changes. The frozen comparisons below are
preserved. All five jobs have now completed; the execution record and links to
measured results follow the plan.

## Runs and fixed settings

| Run | Epoch budget | Architecture | Augmentation | Sampling | Comparison |
| --- | ---: | --- | --- | --- | --- |
| CUDA matched control | 12 | DeepLabV3–MobileNetV3-Large | None | Uniform | Existing 12-epoch Mac run at the same settings |
| Longer control | 40 | DeepLabV3–MobileNetV3-Large | None | Uniform | CUDA matched control: budget only |
| Augmentation candidate | 40 | DeepLabV3–MobileNetV3-Large | Mild | Uniform | Longer control: augmentation only |
| Case sampling candidate | 40 | DeepLabV3–MobileNetV3-Large | None | Case balanced | Longer control: sampling only |
| Larger model candidate | 40 | DeepLabV3–ResNet50 | None | Uniform | Longer control: architecture and its associated pretrained weights |

Every run uses **672 × 384 input, batch size 2, learning rate 0.0003, seed 42,
AdamW, frozen BatchNorm, ImageNet normalization, and balanced cross-entropy**.
Main and auxiliary loss weights remain 1.0 and 0.4. Both architectures start
from their generic Torchvision `COCO_WITH_VOC_LABELS_V1` weights with new
seven-channel heads; none starts from the previously trained anatomy checkpoint.
Keep the same cloud software image and accelerator configuration across runs.
CUDA and MPS are not expected to produce bitwise-identical results.

Balanced class weights retain the existing capped inverse-square-root formula.
Count each selected, resized training mask once, before augmentation or sampling,
and exclude source 255. Record actual weights in each run configuration.
Validation is always unaugmented and visits each selected frame once.

“Mild” means a paired horizontal image/mask flip with probability 0.5, followed
by image-only brightness and contrast factors independently drawn from
[0.9, 1.1]. There is no rotation, crop, mask recoloring, or invented padding.
The image-plane transform does not add an anatomical left/right interpretation.
“Case balanced” assigns each frame weight `1 / selected_frame_count_for_video`,
then samples with replacement for exactly 343 draws per epoch. Videos have
equal expected probability; a particular epoch need not visit every case/frame.

The CLI options are `--architecture deeplabv3_resnet50`, `--augmentation mild`,
and `--sampling case_balanced`. Defaults are MobileNetV3, `none`, and `uniform`.
No candidate combines changes. Keep batch size 2 even on a larger GPU so these
runs answer the stated comparisons; throughput tuning is a separate experiment.

## Data and checkpoint selection

Use the existing Endoscapes manifest without changing case assignments:
**343 train / 75 validation / 74 test frames from 30 / 10 / 10 separate videos**.
Source 255 remains ignored. Validation frame `153_32700`, with unexplained source
ID 7, remains quarantined. No new data acquisition or label reinterpretation
is part of these runs. This is a small, correlated set of surgical frames;
more optimization and a larger backbone do not create new training cases.

Within every run, the existing training code selects `best.pt` using **six-class
foreground macro-IoU on resized validation labels at 672 × 384**. It retains
the first epoch when scores tie. The 40-epoch budget is a maximum optimization
horizon, not a claim that epoch 40 is best. Do not switch this selection rule
mid-experiment or retrospectively select an epoch using a different metric.

Compare the resulting `best.pt` files using `evaluate-original --split val` on
the same 75 frames at the unchanged original annotation grid. Resize logits
before argmax; use raw predictions without confidence or polygon filtering.
The primary comparison is four-class small-anatomy macro-IoU, reported both as
pooled pixels and as equally weighted case scores. The four classes are cystic
duct, cystic artery, cystic plate, and hepatocystic triangle dissection.
Report six-class foreground IoU/Dice and every class's IoU, Dice, precision,
recall, and positive-case count alongside those means. Absent truth with false
positives scores zero; absent truth and prediction is undefined and excluded.

Do not present a candidate as an unqualified winner when pooled and case-equal
scores disagree or improved precision hides worse small-structure recall.
Keep annotation-selected before/after examples, including regressions, and
record checkpoint hashes, manifest identity, software versions, and selected
epochs. Validation is the only selection split for this plan. **Do not evaluate
test images or use the public demo clip to choose settings.** The test split
was inspected in the initial baseline work and is not a fresh blind holdout.

## Budget, interruptions, and interpretation

Forty epochs can show whether the current short run stopped before useful
learning, and gives augmentation a longer opportunity to help. It can also
overfit 30 training cases or repeatedly exploit noise in only 10 validation
cases; the larger model is not guaranteed to improve generalization. Use the
saved training/validation history to explain the selected epoch. This plan
does not add a parameter sweep, extra seeds, or combined changes before results.

Cloud artifact snapshots preserve completed-epoch outputs. **Exact resumption
is not implemented:** checkpoints do not contain optimizer, RNG, or sampler
state. An interrupted job may start a fresh attempt; do not concatenate its
history with an earlier attempt or describe it as seamless continuation.
Preserve separate attempt identities and label any incomplete run explicitly.

After all comparable results are available, select further work from validation
evidence. Combining successful independent changes requires a new documented
experiment rather than an unrecorded modification to this frozen set.

## Execution record — September 13, 2026

All five jobs reached terminal **`JOB_STATE_SUCCEEDED`**, completed their full
epoch budgets and original-grid validation, and have collected artifacts with
verified hashes. They used project `eastwest72hack26bos-501`, `us-central1`, and
one A100 per job. See [CLOUD_RESULTS.md](CLOUD_RESULTS.md) for the complete
metrics, comparability audit, artifact hashes, and local inference timing.

| Run | Vertex custom job ID | Final state |
| --- | --- | --- |
| `cloud-001-control` | `3017512501681061888` | `JOB_STATE_SUCCEEDED` |
| `cloud-002-longer` | `6848949884666511360` | `JOB_STATE_SUCCEEDED` |
| `cloud-003-augmentation` | `2909989060577591296` | `JOB_STATE_SUCCEEDED` |
| `cloud-004-case-balanced` | `6893422930986795008` | `JOB_STATE_SUCCEEDED` |
| `cloud-005-resnet50` | `7671982716568469504` | `JOB_STATE_SUCCEEDED` |

The longer MobileNet control ranks highest on all three aggregate cloud scores.
Its case-equal small-anatomy IoU and artery recall remain below the local demo
reference, so **`small-004-resolution/best.pt` remains the default demo model**.
No combined change, additional job, or test-set evaluation was launched after
these outcomes. The [launch manifest](outputs/cloud-handoff/launch-manifest.json)
preserves the submitted source/data bundles and every filled job configuration;
[CLOUD_TRAINING.md](CLOUD_TRAINING.md) describes artifact retrieval.

## Next data phase — outside these five runs

We have downloaded all **493 public Seg50 masks across all splits**; only
**343 are training examples**. The remaining 76 validation and 74 test masks
retain their split roles, with our one validation quarantine leaving 75 usable.
The larger public releases do not supply the missing ground-truth pixel masks.
The [author repository](https://github.com/CAMMA-public/Endoscapes#contents) calls
Seg201 private, and the [PhysioNet release](https://physionet.org/content/endoscapes-2023/1.0.0/)
confirms that public semantic masks cover the 50-video subset only.

| Release | Labeled frames: train / validation / test | Videos: train / validation / test | Labels available |
| --- | --- | --- | --- |
| Seg50 | 343 / 76 / 74 | 30 / 10 / 10 | Public semantic and instance masks |
| BBox201 | 1,212 / 409 / 312 | 120 / 41 / 40 | Public boxes for the same six classes |
| Seg201 | 1,212 / 409 / 312 | 120 / 41 / 40 | Full segmentation masks remain private |

The [official split definitions](https://arxiv.org/html/2312.12429v1#S3) make
Seg50 a subset of BBox201 within each split. Therefore BBox201 offers
**869 additional box-labeled training frames from 90 additional training
videos** (1,212 − 343 frames; 120 − 30 videos). Using these for segmentation
requires a separate weakly supervised method; boxes are not supplied pixel
masks. No additional acquisition or training on this expansion is included in
the current plan, and validation/test cases must stay outside that training.
