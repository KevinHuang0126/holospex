# Dice iteration results — September 13, 2026

**Do not promote this CE + generalized Dice configuration.** Across three matched seeds, it lowers mean small-anatomy overlap and lowers case-equal small-anatomy IoU and six-class foreground IoU in every pair. Artery overlap improves, but plate overlap and recall regress. This is evidence about the specific tested loss variant and coefficient, not a claim that every Dice loss is inferior.

All six Vertex A100 jobs completed 40 epochs and original-grid validation. Source, data, settings, runtime, frame identities, artifacts and selected checkpoints passed the comparison audit. Input remained **672 × 384**. The new datasets described below were acquired separately and did not enter these runs.

## Matched comparison

CE means the existing balanced cross-entropy. CE + GDL adds the foreground, batch-pooled generalized Dice variant with coefficient 1.0 to both main and auxiliary heads (auxiliary weight 0.4). The [frozen plan](DICE_ITERATION.md) specifies the exact formula, absent-class and ignore policies, checkpoint selection and all fixed settings. Each seed starts fresh from the same generic MobileNetV3 DeepLabV3 weights.

Each run uses 343 training frames from 30 cases. Evaluation uses the same 75 validation frames from 10 cases, at the original 854 × 480 annotation grid. All numbers below are percentages; changes are percentage points. Higher is better. Ranges span the three seeds and are not confidence intervals.

| Metric | CE mean [range] | CE + GDL mean [range] | Mean change |
| --- | ---: | ---: | ---: |
| Small-anatomy IoU, pooled pixels | 25.13 [23.45–26.44] | 24.60 [23.95–25.12] | -0.53 |
| Small-anatomy IoU, equal case weight | 22.42 [21.91–22.79] | 21.49 [21.05–21.89] | -0.94 |
| Six-class foreground IoU | 44.08 [43.13–45.06] | 43.15 [42.87–43.49] | -0.94 |
| Six-class foreground Dice | 55.95 [54.57–57.30] | 55.05 [54.58–55.41] | -0.91 |
| Cystic-artery recall | 21.06 [18.99–23.34] | 22.88 [21.56–24.59] | +1.81 |

Case-equal small-anatomy IoU falls in all three pairs (−1.04, −0.86 and −0.90 pp). Pooled small-anatomy IoU improves in one pair and declines in two. Artery recall improves in two pairs but declines in seed 43. The artery benefit does not offset the broad overlap and plate-recall regressions.

| Seed | Loss | Selected epoch / 40 | Small pooled IoU | Small case-equal IoU | Foreground IoU | Foreground Dice | Artery recall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | CE | 16 | 25.49 | 22.57 | 44.06 | 55.99 | 20.86 |
| 42 | CE + GDL | 20 | 25.12 | 21.52 | 43.49 | 55.41 | 22.48 |
| 43 | CE | 31 | 23.45 | 21.91 | 43.13 | 54.57 | 23.34 |
| 43 | CE + GDL | 23 | 24.71 | 21.05 | 42.87 | 55.15 | 21.56 |
| 44 | CE | 29 | 26.44 | 22.79 | 45.06 | 57.30 | 18.99 |
| 44 | CE + GDL | 30 | 23.95 | 21.89 | 43.08 | 54.58 | 24.59 |

## Per-class means

Each entry is the arithmetic mean of that class metric across three selected checkpoints. IoU, Dice, precision and recall are computed from original-grid pooled pixels within each run; do not reconstruct mean Dice by transforming mean IoU.

| Class | CE IoU | GDL IoU | CE Dice | GDL Dice |
| --- | ---: | ---: | ---: | ---: |
| Gallbladder | 81.20 | 79.37 | 89.62 | 88.49 |
| Cystic duct | 40.07 | 38.85 | 57.20 | 55.87 |
| Cystic artery | 13.95 | 15.86 | 24.48 | 27.37 |
| Cystic plate | 15.39 | 11.14 | 26.40 | 19.96 |
| Triangle dissection | 31.10 | 32.53 | 47.44 | 49.02 |
| Tool | 82.79 | 81.11 | 90.58 | 89.57 |

| Class | CE precision | GDL precision | CE recall | GDL recall |
| --- | ---: | ---: | ---: | ---: |
| Gallbladder | 87.04 | 86.50 | 92.37 | 90.63 |
| Cystic duct | 63.63 | 62.98 | 52.19 | 50.86 |
| Cystic artery | 29.44 | 34.09 | 21.06 | 22.88 |
| Cystic plate | 31.37 | 36.89 | 24.12 | 15.63 |
| Triangle dissection | 47.06 | 52.94 | 48.09 | 45.74 |
| Tool | 88.33 | 86.70 | 92.96 | 92.64 |

The most consequential regression is cystic plate: mean IoU **15.39% → 11.14%** and recall **24.12% → 15.63%**, even though precision rises. Plate IoU/recall decline in two pairs; seed 43 improves both, and the mean decline is strongly affected by seed 44. Duct IoU and recall also decline on average. Artery IoU improves in all three pairs, averaging **13.95% → 15.86%**, and mean triangle IoU improves **31.10% → 32.53%**.

## Diagnostic examples and model decision

The [seed-42 comparison sheet](outputs/dice-handoff/seed42-comparison-small.png) and [full-frame score sidecar](outputs/dice-handoff/seed42-comparison-small.json) use the pre-existing annotation-area selection rule, not prediction-quality selection. In that sheet, before = CE epoch 16; after = CE + GDL epoch 20. Inference sees the full frame; crops are display-only and raw argmax bypasses HUD filtering. The four whole-frame focus IoUs are duct 0.407 → 0.564, artery 0.000 → 0.071, plate 0.000 → 0.000, and triangle 0.460 → 0.000. This preserves failures as well as visible improvements; four examples are not the aggregate result.

The selected demo remains **`ml/outputs/small-004-resolution/best.pt`** (epoch 9), unchanged. Its earlier original-grid validation small IoU is 24.36%, case-equal small IoU 25.73%, and artery recall 34.27%. Every new Dice checkpoint falls below it on case-equal small IoU and artery recall. Keep this older demo reference separate from the matched CUDA objective comparison: it used a different training budget and Mac runtime.

## Additional data actually acquired

| Source | Acquired content | Useful coverage and limitations |
| --- | --- | --- |
| Endoscapes BBox201, additional TRAIN cases | 869 raw images; **852 usable images across 90 new cases**, **3,951 boxes** | 850 usable images have boxes; two have unknown label completeness. Includes 696 duct, 434 artery, 288 plate and 311 triangle boxes, plus gallbladder/tool. These are not pixel masks. |
| Official CholecSeg8k version 11 | **8,080 images with semantic masks**, 17 videos / 101 clips | Maps gallbladder, duct and tools only; 7,331 frames contain a confirmed Holospex target. No artery, plate or triangle labels. Only 248 duct-positive frames, mostly one case. |

Endoscapes acquisition verifies official TRAIN case membership, source identity, box geometry, image dimensions, CRC and SHA-256. Seventeen identical blank frames were quarantined; no usable image hash matches the existing Seg50 images. All 90 new TRAIN cases are separate from existing train and held-out cases. The two empty-annotation images are not implicit negative segmentation examples. See [DATA_EXPANSION.md](DATA_EXPANSION.md) and its visually inspected source-box sheet.

CholecSeg8k retains its original archive, images, masks, license, complete file hashes and an actual pixel-ID audit. Its 17 cases have no Endoscapes overlap listed in the [official CAMMA crosswalk](https://github.com/CAMMA-public/camma_dataset_overlaps) preserved at a pinned commit. Tiny duct labels in two video12 frames are only three pixels total and flagged for review, not counted as meaningful case diversity. A future partial adapter must ignore every pixel outside the confirmed mapping rather than teach unannotated anatomy as background. See [ADDITIONAL_MASK_DATA.md](ADDITIONAL_MASK_DATA.md).

All acquired data and final audit sidecars are also in the private GCP bucket; the tracked [DATA_CATALOG.json](DATA_CATALOG.json) records exact URIs, SHA-256 hashes, object generations, byte counts and integration requirements. The 3.1 GB original CholecSeg8k ZIP has matching local/remote CRC32C. No credentials or dataset payloads were added to Git.

**This does not create a larger fully labeled six-class test set.** The public Seg201 masks remain private. ATLAS-120k is a larger identified, gated follow-up with partial class coverage; it was not downloaded or treated as approved access. Existing validation/test case assignments remain unchanged.

## Next iteration

Prioritize additional small-anatomy supervision over higher resolution. The 90 new Endoscapes TRAIN cases provide the most relevant class coverage. Generate candidate masks from their author boxes in a separate experiment, retain generated-label provenance, review representative cases, and reject uncertain pixels; rectangles must never masquerade as ground-truth segmentation. Use protected held-out cases with reviewed masks to enlarge evaluation coverage. CholecSeg8k can support a separately controlled partial-label pretraining or mixing experiment, with case sampling to prevent its adjacent frames and common structures from overwhelming Endoscapes.

Keep balanced CE as the control. A different Dice variant or coefficient would require a new isolated comparison; this iteration does not justify selecting the current GDL loss or escalating resolution. No additional training was launched after seeing these outcomes.

## Reproduction and audit

All six jobs reached `JOB_STATE_SUCCEEDED`; no run is left active. Each attempt completed training, checkpoint selection and original-grid validation, and all 13 collected artifacts per run passed their recorded SHA-256 and size checks. The [comparison JSON](outputs/dice-handoff/comparison.json) includes all per-seed metrics, paired differences, means/ranges, policy and checkpoint hashes. The [launch manifest](outputs/dice-handoff/launch-manifest.json) retains the source/data bundle inventories and submitted job configurations. A separate [independent audit](outputs/dice-handoff/final-independent-audit.json) reproduced the complete comparison and recomputed pooled, per-video and case-equal metrics from their confusion matrices without discrepancies.

```sh
.venv/bin/python ml/cloud/summarize_dice_results.py \
  --results-root ml/outputs \
  --launch-manifest ml/outputs/dice-handoff/launch-manifest.json \
  --output ml/outputs/dice-handoff/comparison-rerun.json
```

| Run | Selected checkpoint SHA-256 |
| --- | --- |
| `dice-001-ce-s42` | `ecd6eb151eb02fbe00d77fb7a57904b150415ad526a961a0051f8e1c6f7d2244` |
| `dice-002-gdl-s42` | `41fafaf816a0a8e454d3e306dfefacc280a814888a9983109601e533df4f4689` |
| `dice-003-ce-s43` | `ba9faf6a842e3850d7adc693c1d1f4813dc975dea1c3559cb4fbb3bea30befa6` |
| `dice-004-gdl-s43` | `a92f4215b4230ae809cea83663a95605ae3681920e8cb7ebda6f5618d0f29864` |
| `dice-005-ce-s44` | `84082eb4df0a688d26d736d8df320f88fa97baa36e19a77d3cd7a37d9749714d` |
| `dice-006-gdl-s44` | `15173a736e947df18a9eee826ebc87936e9b8fa4f202ebc8171790607aff3f3b` |

Source archive SHA-256: `d5aa297a2d735fd4817c66f54ca32c4568053157a80360c6a5ac193dbca81931`. Data archive SHA-256: `831c81a7d6a8bfb9e9038ce892584f7976dc9e6451026ca60ba11b66a08f56e7`.

The packaged training snapshot passed 135 tests, including tests run after extraction. The final workspace passes **143 ML tests**, including loss mathematics/ignored-pixel gradients, cloud argument forwarding and artifact integrity, data leakage/quarantine checks, and the six-run comparison audit. Validation loss and epoch selection retain the original rules.

Limits: three seeds, only ten repeatedly used validation cases, temporally correlated frames, and nondeterministic CUDA execution. Ranges do not establish statistical significance. The original dataset also contains 22 identical all-black JPEGs across split names; this is documented as a blank-frame quality issue and was kept fixed across this loss comparison. Test was not evaluated in this iteration; its prior inspection means it should not be described as a fresh blind holdout.
