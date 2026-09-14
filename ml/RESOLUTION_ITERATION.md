# Higher-resolution iteration

This iteration tests whether retaining more small-structure pixels improves the
existing model before any teammate-reviewed proposals enter training.

## Frozen runs

| Run | Input | Pixel count vs. 672 × 384 |
| --- | ---: | ---: |
| Existing cloud control | 672 × 384 | 1.00× |
| `resolution-001-896x512` | 896 × 512 | 1.78× |
| `resolution-002-1120x640` | 1120 × 640 | 2.78× |

Both new runs use the same 343 Endoscapes TRAIN frames, 75 usable validation
frames, video split, balanced cross-entropy, DeepLabV3–MobileNetV3-Large,
generic pretrained initialization, 40 epochs, batch size 2, learning rate
0.0003, uniform sampling, no augmentation, and seed 42 as `cloud-002-longer`.
Resolution, run identity, and class weights recalculated from the resized masks
are the only intended training configuration differences. Recalculating the
same weighting formula is part of applying the existing recipe at each grid.
The generalized Dice variant is excluded because the completed
matched comparison reduced the aggregate small-anatomy scores.

Each run selects its checkpoint using resized-grid six-class foreground
macro-IoU, exactly as the existing pipeline does. Final comparison uses the
same original 854 × 480 validation masks and reports small-anatomy pooled IoU,
small-anatomy equal-case IoU, six-class foreground IoU, per-class precision and
recall, selected epoch, runtime, and checkpoint hash. No test images are used.

The main question is whether the extra pixels improve duct, artery, plate, and
triangle validation performance enough to justify greater training and future
inference cost. A higher-resolution winner is not promoted automatically: it
must improve the validation evidence without hiding major per-class recall
regressions. These runs do not use the box-prompted SAM proposals or any
unreviewed masks.

## Completed results

Both Vertex jobs succeeded on one Spot NVIDIA A100-SXM4-40GB worker and all 13
published artifacts from each attempt passed SHA-256 and byte-count verification.
The shared source archive passed 169 ML tests before submission. The comparison
script also verified the shared dataset manifest, run configuration, checkpoint
identity, 40-epoch history, original-grid evaluation policy, and absence of test
evaluation. The complete machine-readable comparison is
[comparison.json](outputs/resolution-handoff/comparison.json).

| Run | Selected epoch | Small pooled IoU | Small equal-case IoU | Foreground IoU | Artery recall | Mean epoch time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 672 × 384 control | 38 | **26.36%** | **25.35%** | 44.77% | 24.63% | 16.79 s |
| 896 × 512 | 35 | 25.46% | 20.82% | 44.80% | 20.02% | 20.28 s |
| 1120 × 640 | 34 | **27.20%** | 24.76% | **45.82%** | **43.87%** | 25.67 s |

The 896 × 512 run does not justify its added cost: pooled small-anatomy IoU fell
0.90 points, equal-case IoU fell 4.53 points, and artery recall fell 4.61 points.
The 1120 × 640 run is useful as a recall-heavy research candidate. Against the
672 × 384 control it improved pooled small-anatomy IoU by 0.84 points,
foreground IoU by 1.05 points, triangle IoU by 9.03 points, and artery recall by
19.24 points. The tradeoff is material: artery precision fell from 30.39% to
17.59%, artery IoU fell 1.39 points, plate IoU fell 4.25 points, and equal-case
small IoU fell 0.59 points. Duct IoU was effectively unchanged.

A warmed batch-one adapter benchmark on one fixed 854 × 480 image measured
median end-to-end CPU latency of 121.89 ms at 672 × 384, 165.28 ms at 896 × 512,
and 227.07 ms at 1120 × 640. This is a relative development-machine benchmark;
it includes image decoding, model inference, geometry export, and schema
validation, but it is not a phone, camera-stream, A100, or glasses benchmark.
The full record is [inference-benchmark.json](outputs/resolution-handoff/inference-benchmark.json).

The [annotation-selected comparison sheet](outputs/resolution-002-1120x640/comparison-small.png)
and [sidecar](outputs/resolution-002-1120x640/comparison-small.json) show one
median-area validation example for each small class, chosen without consulting
either model's predictions. In these four examples the 1120 × 640 model improves
duct IoU 0.392 → 0.618 and artery IoU 0.112 → 0.180, slightly lowers plate IoU
0.252 → 0.232, and sharply lowers triangle IoU 0.335 → 0.063. That variation is
consistent with keeping this checkpoint as a review candidate rather than
treating the aggregate improvement as uniform.

## Decision

Keep `small-004-resolution/best.pt` as the demo checkpoint. It still has the
best equal-case small-anatomy IoU among these candidates and avoids promoting a
model whose artery recall gain comes with substantially more false-positive
area. Retain `resolution-002-1120x640/train/best.pt` for visual review and future
controlled work; reject the 896 × 512 candidate.

This is a single-seed validation comparison and CUDA operations are not fully
deterministic. No test images, public-demo frames, or unreviewed proposal masks
were used for model selection. The next useful model experiment is either a
matched multi-seed confirmation of 1120 × 640 or training with the teammate's
accepted/corrected masks after their review provenance and partial-label policy
are fixed.

## Vertex identities

| Run | Vertex custom job | Completion record |
| --- | --- | --- |
| `resolution-001-896x512` | `6125326098055036928` | `gs://eastwest72hack26bos-501-holospex-ml/runs/resolution-001-896x512/attempts/20260913T151332Z-248027a214614967b96127b5fc8ca5e5/status/completed.json` |
| `resolution-002-1120x640` | `3324087129830588416` | `gs://eastwest72hack26bos-501-holospex-ml/runs/resolution-002-1120x640/attempts/20260913T151331Z-55865213d75349ca8f8f4f229f9c7402/status/completed.json` |
