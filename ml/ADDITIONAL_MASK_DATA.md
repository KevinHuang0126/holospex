# Additional anatomy masks — acquisition and audit

The official **CholecSeg8k version 11** release is downloaded and audited locally
under ignored `ml/data/cholecseg8k/`. This adds actual image/mask assets, but it
has **not been added to training or to the current validation/test splits**.
Its ontology supplies gallbladder, cystic duct and two instrument types. It does
not annotate cystic artery, cystic plate or hepatocystic triangle dissection.
Dataset annotations remain separate from reviewed lesson answers.

## Original release and provenance

- Source: [NEWSLab's official Kaggle release](https://www.kaggle.com/datasets/newslab/cholecseg8k).
- Version 11, dataset ID 901402; official update: September 2, 2021.
- Acquired September 13, 2026 through the anonymous, version-pinned Kaggle API;
  no account, credentials or access-condition flow was used.
- License declaration: **CC BY-NC-SA 4.0**, preserved verbatim in
  [official-metadata.json](data/cholecseg8k/official-metadata.json) and
  [OFFICIAL_DESCRIPTION.md](data/cholecseg8k/OFFICIAL_DESCRIPTION.md).
- The unchanged ZIP is [cholecseg8k-v11.zip](data/cholecseg8k/cholecseg8k-v11.zip),
  **3,106,240,233 bytes**, SHA-256
  `aa0c6f3ded916f5938443571faffcbab408583623093a33e762f1c7b3595e551`.
- [download-provenance.json](data/cholecseg8k/download-provenance.json) records
  the exact download URL, version, timestamps, HTTP metadata and checksums.
  The included standard license text is an unmodified copy of the same
  CC BY-NC-SA 4.0 legal text already stored with Endoscapes; that origin is
  explicit. The CholecSeg8k license grant comes from its own Kaggle declaration.
  The [canonical legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)
  remains linked; its automated text endpoint returned HTTP 403.

Attribute CholecSeg8k to W.-Y. Hong, C.-L. Kao, Y.-H. Kuo, J.-R. Wang,
W.-L. Chang and C.-S. Shih, and retain the underlying Cholec80 attribution to
Twinanda et al. [CholecSeg8k paper](https://arxiv.org/html/2012.12453v1),
[Cholec80 / EndoNet paper](https://arxiv.org/abs/1602.03012).

## What the local audit verified

The complete original release contains **8,080 images**, **8,080 watershed
semantic masks**, **8,080 visualization color masks**, and **8,080 annotation-tool
masks**: 32,320 original files in total. All files were read fully through the ZIP
CRC checker during extraction, and each extracted file has a recorded SHA-256.
Original files are preserved under [raw-v11/](data/cholecseg8k/raw-v11/).

There are **17 source videos and 101 clips**, each containing 80 images. Every
image is 854 × 480 and has all three corresponding masks; all watershed masks
match the image dimensions. No duplicate video/frame identities were found.
There are 8,076 RGB watershed masks and four RGBA masks; their RGB channels agree
everywhere, and the four alpha channels are entirely opaque.

Full evidence: [audit.json](data/cholecseg8k/audit.json),
[frame-index.jsonl](data/cholecseg8k/frame-index.jsonl), and
[archive-file-index.jsonl](data/cholecseg8k/archive-file-index.jsonl).

| Confirmed source label | Actual stored channel value | Holospex channel / ID | Frames containing pixels |
| --- | ---: | --- | ---: |
| Gallbladder | 22 | 1 / `gallbladder` | 6,861 |
| Cystic duct | 25 | 2 / `cystic_duct` | 248 |
| Grasper | 31 | 6 / `tool` | 6,020 |
| L-hook electrocautery | 32 | 6 / `tool` | 2,254 |

Tool rows overlap: their union is 6,438 frames. A total of **7,331 frames** contain
at least one confirmed Holospex target; the remaining 749 would have entirely
ignored targets under the conservative mapping below.

The cystic-duct count is especially concentrated: video01 contributes 244 frames
and 1,155,391 pixels; video17 contributes two frames and 66,817 pixels; video12
contributes two frames containing only **three pixels total**. The latter is not
meaningful new case coverage. These are temporally correlated annotations, not
248 independent procedures or evidence of segmentation quality.

## Required interpretation before training

The official description calls the repeated codes RGB hex codes. The actual
stored channel values are **decimal annotation IDs**: cystic duct is 25, not
hexadecimal `0x25` (= 37), and not contiguous class index 8. Source background is
50. The observed values are `0, 5, 11, 12, 13, 21, 22, 23, 24, 25, 31, 32, 33,
50, 255`. Do not reuse the Endoscapes source-ID mapping.

Values 0 and 255 are not defined as anatomical categories by the source table.
There are 125 zero-valued pixels across 52 masks, and 24,557,849 value-255 pixels
across 8,026 masks. Preserve both as ignored; do not silently assign a class.

For an initial partial-supervision adapter, map only confirmed positive source
IDs **22, 25, 31 and 32**. Set **every other pixel**, including source background,
other tissues and unknown values, to ignore index 255. Unannotated cystic artery,
plate and triangle pixels must not become background negatives. Keep fully
supervised Endoscapes batches for the complete ontology and background learning.
Exclude all-ignored external frames explicitly. No converted targets, mixed
training manifest, new split or training run were created by this acquisition.

## Case overlap checked against the official crosswalk

**The acquired 17 CholecSeg8k cases have no Endoscapes case overlap listed in
CAMMA's official mapping.** This supersedes the initial unverified status.
The [Endoscapes repository](https://github.com/CAMMA-public/Endoscapes) links the
authors' [dataset-overlap analysis](https://github.com/CAMMA-public/camma_dataset_overlaps),
which was preserved at commit `8347b9f4cb02ebe739747903e6eada272ee9d25e`.

The local audit joins Cholec80 IDs to original case IDs, then to Endoscapes
splits/public IDs. Across all Cholec80 videos, the six shared cases are:
Cholec80 67/68/70/71/72 → Endoscapes train 1/2/3/4/7, respectively;
Cholec80 66 → Endoscapes validation 121. None is in the acquired set:
`1, 9, 12, 17, 18, 20, 24, 25, 26, 27, 28, 35, 37, 43, 48, 52, 55`.
The CholecSeg8k release explicitly identifies its top-level folder names as the
original Cholec80 video filenames. This is an author-provided identity join,
not an inference that equal or different numeric names identify cases.

Both datasets involve Strasbourg institutions, so hospital names and publication
dates alone cannot establish disjointness. The crosswalk is the stronger evidence.
Keep identities namespaced. This check does not independently audit image
duplicates or establish a new independent test set; no data was merged.
Evidence: [overlap audit](data/cholecseg8k/overlap-evidence/cholecseg8k-overlap-audit.json),
[original mapping files and provenance](data/cholecseg8k/overlap-evidence/provenance.json).

## Three-frame source-label visual audit

The [contact sheet](outputs/cholecseg8k-source-audit/source-label-contact-sheet.png)
was generated with Pillow from unchanged originals and watershed IDs, then
visually inspected. It shows source images, overlays, focused crops and an
explicit decimal-ID key. Selection rules, pixel counts, crop coordinates and
input SHA-256 values are in the
[visual audit](data/cholecseg8k/source-label-visual-audit.json).

| Source frame | Selection / observed annotation |
| --- | --- |
| `video01/frame_16465_endo.png` | Median duct area among video01 frames with ≥64 duct pixels; 5,012 pixels form an extended annotated region. |
| `video12/frame_15851_endo.png` | Two diagonal duct pixels at `(286,397)` and `(287,398)`, surrounded by source fat ID 12. |
| `video17/frame_1599_endo.png` | Median gallbladder area among frames with gallbladder and tool; 31,327 gallbladder pixels and 2,618 grasper pixels. |

The other video12 duct-positive mask, frame 15856, has one pixel at `(236,401)`,
also surrounded by source fat ID 12. These isolated spots are **candidate
annotation artifacts requiring source/expert review**; visual inspection cannot
establish a correct replacement label. They do not provide a usable duct shape.
No source masks were changed, and no clinical correctness is claimed for any
of the three examples.

## Private cloud copies

The original version-11 ZIP and a separate archive containing the final metadata,
file hashes, license, ontology and crosswalk audits are uploaded to the private
hackathon bucket. The [cloud data catalog](DATA_CATALOG.json)
records their immutable URIs, SHA-256 hashes, generations and sizes. The original
3,106,240,233-byte ZIP's local and remote CRC32C values both equal `e9+6Pw==`.
No external data was added to the current Dice experiment or its held-out splits.

## Larger follow-up: ATLAS-120k

The [official ATLAS release](https://huggingface.co/datasets/TimJaspersTue/ATLAS-120k)
reports 121,018 masks from 100 videos, including cystic duct and cystic plate.
Access remains gated, with contact sharing and acceptance required; no account
action or ATLAS download was performed. Most images require separately obtained
source videos. If acquired, use original IDs: 13 = cystic duct, 14 = gallbladder,
17 = cystic plate. The authors' consolidated taxonomy merges multiple ducts;
generic artery is not cystic artery, and there is no triangle-dissection class.
[Official class definitions](https://github.com/TimJaspers0801/ATLAS/blob/main/atlas120k_tools/classes.py).
