# First returned anatomy-mask review — September 13, 2026

The returned pilot records decisions for all **52 candidates in 20 images**.
There are **36 accepted unchanged, 13 edited and approved, 2 needs-expert, and
1 rejected**. No candidates are pending or missing. The reviewer declared
anatomy scope. This is a human annotation record, not a model-accuracy evaluation.

| Anatomy | Accepted unchanged | Edited and approved | Excluded | Imported masks |
| --- | ---: | ---: | ---: | ---: |
| Cystic artery | 11 | 3 | 2 needs expert | 14 |
| Cystic duct | 14 | 1 | 1 rejected | 15 |
| Cystic plate | 7 | 2 | 0 | 9 |
| Hepatocystic triangle dissection | 4 | 7 | 0 | 11 |
| Total | 36 | 13 | 3 | 49 |

## Validated import

The review passed the existing strict validator against the exact original
bundle, image files, proposal identities, dimensions, masks and decision rules.
The importer preserved the submitted JSON verbatim and produced **49 separate
approved masks across 18 images/cases**, plus a receipt and per-mask provenance.
Every imported PNG was decoded and compared with the review's pixels and SHA.
The three excluded candidates produced no approved mask. Original proposals,
reviewer decisions, and existing datasets were not changed.

Local artifacts, including the reviewer name and full notes, remain outside Git:

- [Import receipt](outputs/review-imports/pilot-001-binh-20260913T173030Z-79895ae26847/receipt.json)
- [Detailed audit](outputs/review-imports/pilot-001-binh-20260913T173030Z-79895ae26847/analysis/REPORT.md)
- [Machine-readable audit](outputs/review-imports/pilot-001-binh-20260913T173030Z-79895ae26847/analysis/audit.json)
- [Before/after examples](outputs/review-imports/pilot-001-binh-20260913T173030Z-79895ae26847/analysis/correction-examples.jpg)

Source review SHA-256:
`79895ae268475547403c9d06c8f336a7cf7d60a4bc99d84886ea4a4738481282`.
Original bundle SHA-256:
`8582a85d10a69313fb4672e3580e63f038d01bd6f376f79e7544cb42e0cc2505`.

## What the review reveals

Seven of eleven triangle-dissection proposals were edited; each of those seven
was expanded. The reviewer also corrected masks covering adjacent structures.
Across all thirteen edited masks, **52,584 pixels were added and 20,823 removed**.
Some changes substantially replace the original proposal. These are concrete
annotation corrections; they do not yet demonstrate better model performance.

The project lead confirmed that the notes on `22_41925` and `95_33000` describe
the original errors and that the reviewer already made the fixes. Both approved
edits are included. This clarification is recorded separately from the original
review in the [resolution record](outputs/review-imports/pilot-001-binh-20260913T173030Z-79895ae26847/resolutions/20260913-notes-fixed-ignore-overlap.json).

## Training targets prepared

- All 20 pilot cases are additional TRAIN cases, separate from the existing
  Seg50 train/validation/test cases. Keep validation and test unchanged.
- **15 cross-class mask pairs overlap across 10 images**, with **5,457 distinct
  pixels** assigned to multiple approved classes. A single-class segmentation
  target now ignores these conflicting pixels, following the lead's selected
  policy; no class wins by mask write order. Same-class overlaps are unioned.
- Imported candidate PNGs encode **255 = reviewed foreground; 0 = unknown**.
  These are partial positive annotations, not Endoscapes dense targets. The
  existing Endoscapes loader must not consume these PNGs unchanged. The new
  [target converter](PARTIAL_TRAINING.md) maps uncontested pixels to the explicit
  anatomy source IDs and sets uncovered/conflicting pixels to ignore ID 255.
- Rejected and needs-expert records remain excluded. A reviewer's named
  approval records their judgment; file validation does not assess anatomy.

The [prepared manifest](outputs/reviewed-partial-001/manifest.json) adds 18 partial
training images to the original 343, yielding **361 train / 75 validation / 74
test** images. It retains 323,094 reviewed foreground pixels (98.34%) and leaves
the original dataset, source review and holdout records unchanged. All 181 ML
tests pass; actual target hashes, loader resizing at four resolutions and zero
ignored-pixel CE/Dice gradients were independently checked. See the
[verification record](outputs/review-imports/pilot-001-binh-20260913T173030Z-79895ae26847/analysis/preparation-verification.json).

The subsequent [six-run training comparison](REVIEWED_DATA_RESULTS.md) is now
complete: all three paired seeds improved small-anatomy pooled and equal-case
IoU, supporting another bounded review batch while plate remains a weakness.
The original import receipt and preparation snapshots remain unchanged records
of their earlier steps.
