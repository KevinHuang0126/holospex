# Training targets from reviewed partial masks

For later batches, the resolution record may include `excludedCandidates`, an
array of `{candidateId, reason}` objects. These are explicit training exclusions
of otherwise approved masks; they do not change the original review decision
or import receipt. The converter preserves original eligibility separately from
training eligibility and records reasons in manifest, summary and image
provenance. Unknown, duplicate, nonapproved or malformed exclusions fail.
The [second batch](REVIEW_BATCH_002.md) exercises this path on top of the first
pilot manifest, preserving the inherited sample prefix and holdouts.

The project lead confirmed that the notes on `22_41925` and `95_33000` described
the original mask errors, and that the reviewer had already made the fixes.
Both approved edits remain included. The lead also selected the conservative
overlap policy: ignore pixels assigned to multiple anatomy classes.

This is a derived dataset preparation step. It preserves the original review,
source proposals and base Seg50 dataset. It does not change the model architecture,
start training, or supply lesson answers.

## Exact label policy

1. Include only anatomy-scope accepted or edited candidates from the validated
   review. Rejected, needs-expert, missing and pending candidates supply no label.
2. Combine masks with the same anatomy class using their union.
3. A pixel covered by exactly one eligible class receives that class's explicit
   Endoscapes source ID from the base manifest.
4. A pixel covered by two or more different eligible classes is **255: ignore**.
   Class order never decides a winner.
5. A pixel covered by no eligible mask is also **255: ignore**. It does not become
   background. This includes unreviewed surroundings and excluded candidates.
6. Images with no uncontested positive pixels are omitted from the added training
   samples. All reviewed cases must be new official TRAIN cases, separate from
   every existing Seg50 train/validation/test case.

| Anatomy | Derived PNG source ID | Model target index |
| --- | ---: | ---: |
| Cystic duct | 4 | 2 |
| Cystic artery | 3 | 3 |
| Cystic plate | 1 | 4 |
| Hepatocystic triangle dissection | 2 | 5 |
| Unknown or conflicting | 255 | 255 (ignored) |

These are the current base-manifest values, not an assumed universal mapping.
The converter reads and validates that manifest. The imported per-candidate PNGs
have a different meaning (`255 = positive`, `0 = unknown`) and must never be
passed directly to the Endoscapes training loader.

## Reproduce the prepared dataset

Run from the repository root, choosing a fresh output directory:

```sh
.venv/bin/python ml/review/prepare_training.py \
  --bundle ml/outputs/holospex-mask-review-pilot-001/bundle.json \
  --review ml/outputs/review-imports/pilot-001-binh-20260913T173030Z-79895ae26847/review.json \
  --base-manifest ml/outputs/endoscapes-manifest.json \
  --resolution-record ml/outputs/review-imports/pilot-001-binh-20260913T173030Z-79895ae26847/resolutions/20260913-notes-fixed-ignore-overlap.json \
  --output-dir ml/outputs/reviewed-partial-001
```

The resolution record binds the lead's confirmation and overlap choice to exact
bundle, review and base-manifest SHA-256 values. It is separate from the original
reviewer's decision file; no later clarification is attributed to the reviewer
as an edited or re-exported decision.

The prepared `manifest.json` is for the existing Python segmentation loader.
It combines the original fully labeled samples with additional partial samples,
preserving the original validation/test records and their order. Its local file
paths must be staged/rebased with integrity checks for a later cloud experiment.
The default Vertex entry point regenerates the original Seg50 manifest.
The optional verified prepared-package path now stages this combined manifest;
see [the training comparison](REVIEWED_DATA_ITERATION.md) and
[cloud instructions](CLOUD_TRAINING.md#reviewed-partial-training-package).

## Verified native-resolution reference

The 49 approved candidate masks across 18 images cover 328,551 distinct foreground
pixels. Removing 5,457 conflicting pixels retains **323,094 pixels (98.34%)**.

| Anatomy | Retained pixels | Fraction removed from this class |
| --- | ---: | ---: |
| Cystic artery | 52,025 | 6.39% |
| Cystic duct | 142,239 | 0.03% |
| Cystic plate | 61,717 | 4.59% |
| Hepatocystic triangle dissection | 67,113 | 6.09% |

Overall retention hides larger local changes: the triangle mask in `29_40050`
loses 44.3% of its positive pixels. Preserve these cases for possible later
annotation refinement; ignoring their ambiguous pixels does not decide anatomy.

An independent RLE-based reference check found that all 18 usable images retain
every represented class at 448 × 256, 672 × 384, 896 × 512 and 1120 × 640 using
the loader's nearest-neighbor mask resizing. No background label is introduced.
The [reference artifact](outputs/review-imports/pilot-001-binh-20260913T173030Z-79895ae26847/analysis/preparation-reference.json)
records native pixel hashes and per-image counts at each resolution.

These partial positives should be used alongside the original fully labeled
training images, with the same untouched validation split for comparison. Only
a subsequent controlled training/evaluation run can establish a model benefit.

## Prepared output and exercised checks

- [Combined training manifest](outputs/reviewed-partial-001/manifest.json):
  361 train / 75 validation / 74 test samples; 18 new TRAIN cases.
- [Preparation summary](outputs/reviewed-partial-001/summary.json): source
  bindings, class counts, omitted cases, per-image provenance and artifact hashes.
- [Independent verification](outputs/review-imports/pilot-001-binh-20260913T173030Z-79895ae26847/analysis/preparation-verification.json):
  exact native target agreement with the separate RLE reference; actual loader
  targets at all four sizes; zero CE/Dice logit gradients at ignored pixels at
  672 × 384 and 1120 × 640; all 984 original base files unchanged.

The full ML suite passes **181 tests**, including 12 focused converter/loader
tests. They cover mismatched source/model class IDs, conflict and unknown
handling, source/resolution identity, split leakage, held-out preservation,
same-class union, all-ignore targets, resize loss of a synthetic tiny label,
and safe cleanup/no overwrite on failure. No real model training was needed for
these preparation checks. Subsequent cloud training is recorded separately in
[REVIEWED_DATA_ITERATION.md](REVIEWED_DATA_ITERATION.md); preparation snapshots
remain unchanged.
