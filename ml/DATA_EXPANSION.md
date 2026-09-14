# Endoscapes training data expansion

Acquisition is complete: the public Endoscapes-BBox201 training split supplied
**869 additional frames from 90 additional training cases**, beyond the 343
Seg50 training frames from 30 cases. After quarantining 17 repeated blank
images, the separate usable manifest contains **852 unique frames from all 90
new cases and 3,951 boxes**. Files are in
`data/endoscapes-bbox-train-expansion/`. This data is not connected to the dense
segmentation training loader or the current Dice-loss comparison.

These labels are **bounding boxes**, not pixel masks. The authors release
segmentation for Seg50 only; the larger Seg201 pixel-mask dataset is private.
The [official release](https://github.com/CAMMA-public/Endoscapes) documents that
Seg50 is a strict subset of BBox201, and identifies the training/validation/test
case lists. This expansion uses only the author's additional training cases.

## Acquired label content

Source training COCO metadata has 1,212 frames. Exactly 343 overlap the existing
Seg50 training set. The remaining 869 frames have 3,951 box annotations:

| Structure | Additional boxes |
| --- | ---: |
| Cystic duct | 696 |
| Cystic artery | 434 |
| Cystic plate | 288 |
| Hepatocystic triangle dissection | 311 |
| Gallbladder | 839 |
| Tool | 1,383 |

The raw source includes 850 frames with at least one box and **19 frames with
empty annotation lists**. All 17 quarantined frames have empty annotation lists,
so the usable set retains all 3,951 boxes and two frames with empty annotations:
`42_44900.jpg` and `42_45650.jpg`. Exact filenames are retained in each manifest's
`summary.emptyAnnotationFrames`.
Empty annotations do not establish background-only imagery. The manifest marks
`labelCompleteness: "not_established"` and `pixelSupervision: "none"` explicitly.
A box identifies an extent; it does not identify which enclosed pixels belong
to the structure. The author's `area` field is preserved as `sourceArea`; it is
not necessarily the rectangle's width times height.

## Files and schema

- `source/LICENSE`, `source/README.md`: unchanged terms and author documentation.
- `source/train/annotation_coco.json`: unchanged source training annotations.
- `source/*_vids.txt`: original full and Seg50 case lists for split verification.
- `plan.json`: selection and leakage audit before image acquisition.
- `images/VIDEO_FRAME.jpg`: additional original training JPEGs only.
- `quarantine/images/VIDEO_FRAME.jpg`: raw duplicates or uniformly black frames.
- `acquired-manifest.json`: all 869 source frames, including explicit quarantine.
- `manifest.json`: usable frames only, with per-image SHA-256/size and box records.
- `download-TIMESTAMP.json`: immutable source/transfer/hash provenance for a run.

The completed [download provenance](data/endoscapes-bbox-train-expansion/download-20260913T043824462874Z.json)
records 95,208,039 raw JPEG bytes and 97,294,963 selected source bytes including
metadata. The successful resumed run transferred 134,532,487 bytes, including
range/ZIP overhead; this excludes the earlier interrupted attempt and initial
metadata inspection. The usable JPEGs occupy 95,087,220 bytes. Every one of the
869 final image files was independently reread and checked against its recorded
size and SHA-256 after acquisition.

Final manifest SHA-256 values:

- [Usable manifest](data/endoscapes-bbox-train-expansion/manifest.json):
  `5821ae705ca623955600f98064cf3d33942ea00cb39f4e235dbdd810d1f8c224`.
- [Complete acquired inventory](data/endoscapes-bbox-train-expansion/acquired-manifest.json):
  `993b82127a4e7ecff558e367181bec763d53a550c3bbfa15fc73e6af7bac8f2c`.

`manifest.json` is a separate data manifest, not the frontend `FrameResult`
contract and not a replacement for the existing Seg50 manifest. Every image
record declares `split: "train"`, `annotationType: "bounding_box"`, original
width/height, source image ID, case ID, and frame number. The image path is
relative to the expansion directory. Boxes use original-image pixel coordinates
in `[x, y, width, height]` order, with the origin at the top left.

For example, source frame `1_30100.jpg` is 854 × 480 and includes this box:

```json
{
  "annotationId": 10301001,
  "bboxXYWH": [307.0, 294.0, 57.0, 122.0],
  "sourceCategoryId": 3,
  "structureId": "cystic_artery",
  "iscrowd": 0,
  "sourceArea": 2757
}
```

The original source category IDs are preserved. Canonical `structureId` values
reuse the existing dataset name mapping; source `calot_triangle` maps to
`hepatocystic_triangle_dissection`. No new class ordering or PNG label mapping is
introduced. Raw source CVS tags remain in the source metadata and are not
converted into reviewed lesson answers.

## Acquisition and leakage checks

Install the existing data dependencies, then run the independent module:

```sh
.venv/bin/python -m holospex_ml.box_data \
  --seg50-root ml/data/endoscapes \
  --output-dir ml/data/endoscapes-bbox-train-expansion --download
```

Omit `--download` to inspect metadata and write the plan without acquiring
imagery. This still transfers the ZIP central directory. Importing the module
has no network side effects. Existing matching files are verified and reused;
differing destination files are never overwritten.

The downloader reuses the bounded HTTP-range and ZIP validation functions in
`download.py`, selecting only the 869 canonical JPEG members from the author's
[archive](https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip).
It verifies archive ETag/size, ranged responses, ZIP member CRC/size, JPEG format
and dimensions, and per-file SHA-256. Provenance records actual transferred and
selected bytes plus each member's archive path, checksum, and local path. It
does not fetch the entire approximately 6.29 GB ZIP.

Before imagery is selected, the planner checks that the author's full and Seg50
case lists match the existing local release, all splits are disjoint, Seg50
train is an exact subset of BBox201 train, and the expansion contains only new
training cases. All 81 full-release validation/test cases remain excluded.
The geometry check rejects out-of-image, empty, or nonfinite boxes. Unexpected
segmentation payloads or changed release counts require reinspection.

Image hashes are also compared against all 493 existing Seg50 JPEGs, including
held-out images. Every exact-content match or uniformly black frame goes into
quarantine and is excluded from the usable manifest. Its raw source frame ID,
box labels, hash, exclusion reasons, and existing matching frame/split identities
remain in the acquired inventory and audit. Reading local held-out image hashes
is a leakage audit, not an evaluation or a new download. The existing
quarantined Seg50 frame and ignore-label policy remain unchanged.

The audit found one pre-existing group of **22 identical all-black Seg50 JPEGs**:
5 train, 2 validation, and 15 test filenames, each 7,107 bytes, SHA-256
`4fb2c8b95d24c9345e9771cc4e2263dceba6f6c60029b7f8c92ddb9629230e6f`.
Visual inspection and decoded RGB extrema confirmed every pixel is zero.
This is a repeated blank-frame quality issue, not evidence that clinical imagery
from the same case crosses the official splits. The expansion excludes matches;
the original Seg50 samples and active paired experiments are not modified.

The completed audit found **zero usable image-hash matches with existing Seg50
data**, zero overlap with held-out case IDs, and 852 distinct usable image
hashes. All 17 quarantined expansion JPEGs match the same pre-existing black
image hash above. Their matching original filenames and splits remain in
`leakageAudit.quarantinedFrames` and the complete acquired inventory.

## Visual source-box audit

The [four-frame audit sheet](outputs/bbox-expansion-audit/source-box-audit.png)
shows original author boxes in the canonical anatomy colors, with frame and
annotation IDs. Each panel displays only its named class; other source boxes
are omitted for readability. Selected frames are `103_11975.jpg` (artery),
`92_104675.jpg` (duct), `118_117225.jpg` (plate), and `68_46975.jpg` (triangle),
all from new training cases. Exact source paths, hashes, boxes, and the final
independent file verification are in
[verification.json](outputs/bbox-expansion-audit/verification.json). The ignored
`outputs/bbox-expansion-audit/render_audit.py` reproduces the sheet.

Visual inspection found no obvious coordinate displacement, scaling mismatch,
blank imagery, or clipped annotation rectangle in these four examples. The
triangle box is narrow (46 × 63 original pixels); the plate and duct examples
include glare and instrument-adjacent tissue. This verifies that the source
coordinates render on their matching images; it does not establish clinical
label correctness or that every visible structure is annotated.

## Use and next steps

The complete raw acquisition, usable manifest, quarantine inventory, source
license and checksum provenance are also stored in the private hackathon GCP
bucket. [Cloud data catalog](DATA_CATALOG.json)
records the immutable object URI, SHA-256, object generation and byte count.
The data archive SHA-256 is
`62c3e97a7f55eac13ff8709152d7b42ad06d32a9454956cf9a8ce2e4d8e93abd`.
This storage handoff does not convert boxes into masks or enroll them in training.

This data is suitable for a separate detection experiment or a documented
box-supervised segmentation method. It cannot be passed to the current dense
pixel-loss pipeline as if rectangles were segmentation truth. A future
box-prompted pseudo-mask experiment needs its own generated-label provenance,
unknown-pixel policy, validation, and loss weighting. Preserve generated labels
as predictions; do not relabel them as author masks or reviewed answers.

The first [teammate review pilot](MASK_REVIEW.md) is now available: 20 images from
20 distinct new TRAIN cases, with 52 SAM 2.1 proposals generated from the original
small-anatomy boxes. Selection was frozen before inference and was not filtered
by model quality score. Original boxes/images, proposal pixels and reviewer edits
are preserved separately. The portable editor exports named review decisions;
the importer accepts only anatomy-scope approved masks as partial positives.
The [first returned review](REVIEW_PILOT_RESULTS.md) covers all 52 candidates;
49 approved masks were imported with provenance. The lead confirmed the two
flagged edits were complete and selected ignore-overlap targets. The
[prepared expansion](PARTIAL_TRAINING.md) adds 18 partial training images while
preserving validation/test data. The [six-run training comparison](REVIEWED_DATA_RESULTS.md)
has now completed: three paired seeds improved small pooled IoU by an average
2.83 points and equal-case IoU by 2.06 points, with plate precision still weak.
The remaining bounding-box-only acquisition is not enrolled automatically.

The source release is **CC BY-NC-SA 4.0**, for non-commercial scientific research
under the author's terms. Keep its license, attribution, and source records with
any derived dataset; store acquired images/annotations outside Git. The source
README supplies the Endoscapes benchmark and latent-graph paper citations.
No external access request or credentialed/private Seg201 download is included.

Nine new tests cover split leakage, category/identity drift, malformed box
geometry, explicit empty annotations, box-versus-mask semantics, and content
quarantine. The full ML suite passed **137 tests** after implementation; results
are recorded in `outputs/bbox-expansion-tests.log`.
