# Teammate handoff: candidate anatomy-mask review

The second batch has now been returned and validated: 141 decisions, 121
approved candidate masks, and 119 enrolled for training after two explicit lead
exclusions. The combined dataset has 407 train / 75 val / 74 test images.
See [batch002 import and training record](REVIEW_BATCH_002.md). The description
below is retained as the original batch handoff record.

The next review batch is ready: **50 images from 25 new TRAIN cases, with 141
model-generated candidate masks**: 45 ducts, 34 arteries, 30 plates and 32
triangle-dissection regions. Each case contributes two images separated by at
least 750 source frames. All 20 first-pilot cases, original Seg50 cases and
held-out cases are excluded. This batch follows the [reviewed-data training
results](REVIEWED_DATA_RESULTS.md); duct and plate boundaries deserve particular
attention. At handoff these proposals had zero review decisions and had not
entered training. The model selection at that time was unchanged.

The original pilot contained 20 images from 20 cases and 52 candidates:
16 arteries, 16 ducts, 9 plates and 11 triangle-dissection regions.

The first pilot review has now been returned: [results and validated import](REVIEW_PILOT_RESULTS.md).
All 52 decisions are recorded and 49 approved partial masks are preserved.
The workflow below applies to either batch; use each batch's exact bundle and
review export together. Returned reviews remain separate, attributed artifacts.

The original boxes come from the authors. A generic SAM 2.1 large model proposes
pixel masks using those boxes. It does not identify or medically verify the
anatomy class: the class comes from the source box label. The proposed boundaries
remain model-generated until a reviewer records a decision.

## What to give your teammate

Give them the completed ZIP at
[holospex-mask-review-batch-002.zip](outputs/holospex-mask-review-batch-002.zip).
The ZIP includes **START-HERE.txt** with complete nontechnical instructions.
They extract it and open `review.html` in a current desktop browser. It includes
all images and masks and needs no Python, GCP access, login, or internet.
The embedded data is an offline copy, not a live shared review service.

Suggested message to forward with the ZIP:

> Please review this new 50-image anatomy batch. Extract the ZIP, read
> START-HERE.txt, and open review.html. Enter your name and choose
> "Anatomy — assess correctness". For each
> structure, compare the colored mask with the original image and source box.
> Accept a good mask; brush/erase to correct one and approve the edit; reject a
> wrong proposal or mark Needs expert with a note. Export review JSON regularly
> and send the exported file back to me. You can import that file to resume.
> Please flag uncertain anatomy instead of guessing. These are segmentation
> annotations, separate from lesson answers or CVS assessment.

The user confirmed that the intended teammate can assess the anatomy. The tool
also supports technical review, which remains separate from anatomy-approved
supervision. It records the reviewer's name, scope, decision, notes, time and
original proposal identity; a name/scope declaration is not credential verification.

## Review procedure

1. Inspect the source image with the mask hidden, then toggle the overlay and
   original box. The box is only a location hint, not a pixel boundary.
2. Check the structure identity and visible extent. Look for spill into adjacent
   tissue, missing branches, disconnected visible parts, and obscured edges.
3. Zoom and adjust the mask with brush/erase, using undo as needed. Corrections
   remain full-resolution raster masks so holes and separate components survive.
4. Record a decision explicitly. Changes after acceptance return the candidate
   to pending. Rejected/uncertain candidates need a short note.
5. Export JSON after a few decisions and before closing. Browser storage is a
   convenience; the exported file is the portable backup. Import preserves the
   original reviewer and verifies the batch and candidate identities.

A useful first checkpoint is five images: send that partial export to the lead
so the proposal quality and review effort can be assessed before finishing all
141 candidates. No minimum approval rate is expected, and rejection is useful
feedback. Do not approve an uncertain mask to finish the queue.

The mask score is SAM's predicted overlap quality, not confidence that the named
anatomy is correct. Geometry warnings are review cues, not correctness labels.
The review batches omit gallbladder/tools and do not assess complete per-frame coverage.
Unreviewed surrounding pixels remain unknown.

## Lead: validate the returned file

Keep the original bundle/images unchanged. The tool validates the returned review
against the exact bundle SHA, image hashes, proposal hashes, dimensions, RLE,
reviewer attribution and decision semantics before writing any import output.
The handoff contract is documented in [review/FORMAT.md](review/FORMAT.md).

```sh
.venv/bin/python ml/review/review_io.py validate \
  --bundle ml/outputs/holospex-mask-review-batch-002/bundle.json \
  --review /path/to/returned-review.json

.venv/bin/python ml/review/review_io.py import-review \
  --bundle ml/outputs/holospex-mask-review-batch-002/bundle.json \
  --review /path/to/returned-review.json \
  --output-dir ml/outputs/review-imports/teammate-batch-002
```

The import writes a receipt and separate masks only for anatomy-scope accepted
or edited candidates. It preserves the original proposals and reviewer evidence.
Technical-only, rejected, uncertain, missing and pending records do not supply
eligible anatomy masks. An empty accepted mask is invalid.

Separate accepted masks can overlap; the importer does not silently choose a
winning class or merge them into a dense target. Partial coverage/conflicts need
an explicit training-data policy before a later experiment. No import enrolls
data in training or evaluation, and new TRAIN cases do not become test cases.

## Reproduce a portable package

The generation record, exact source selection and immutable model provenance
live with `outputs/review-batch-002/` and `outputs/review-batch-002-cloud/`. The proposal
model's default internal image preprocessing is independent of Holospex training;
this is mask-proposal inference, not higher-resolution segmentation training.

```sh
.venv/bin/python ml/review/package_review.py \
  --bundle ml/outputs/review-batch-002/bundle.json \
  --output-dir ml/outputs/holospex-mask-review-batch-002 \
  --license ml/outputs/review-batch-002/provenance/Endoscapes_LICENSE.txt \
  --license ml/outputs/review-batch-002/provenance/SAM2_LICENSE.txt
```

Use a fresh output directory for another package. The single HTML embeds its
images and proposals; the ZIP also contains the immutable bundle, original JPEGs,
license texts, instructions and a checksum inventory. Raw data/reviews stay out
of Git; the tooling and handoff documentation are tracked source.

Sources: [official SAM 2 repository](https://github.com/facebookresearch/sam2),
[Endoscapes release](https://github.com/CAMMA-public/Endoscapes),
[local box-data audit](DATA_EXPANSION.md). Preserve the Endoscapes CC BY-NC-SA 4.0
terms and attribution with derived artifacts, and SAM's Apache-2.0 notice.

## Batch 002 generation and verification record

Vertex job `5340186834892750848` generated the new proposals on one Spot A100
using the same pinned SAM 2.1 large model as the original pilot. Inference took
12.46 seconds, excluding startup. Selection uses source-box class,
area and geometry strata; it does not rank by model confidence or reviewer
labels. The [selection audit](outputs/review-batch-002-cloud/independent-selection-audit.json)
verifies 50 original images and every selected source box.

Bundle SHA-256: `91ad669b5a0b5225468908e25a1b344808571234ad3a36a928798af54e419586`.
The [handoff inventory](outputs/holospex-mask-review-batch-002/handoff-manifest.json)
records all packaged file hashes and zero created review decisions. The new
START-HERE.txt describes exact editor labels, explicit approval after any note
or mask changes, save/resume, and returning only the latest exported JSON.
The original ZIP, bundle and returned review remain unchanged.

Validation: 211 Python ML tests and 12 JavaScript review-core tests pass.
See `outputs/review-batch-002-cloud/` for cloud, selection, generation and ZIP
audits. Current browser verification is recorded there separately from the
original pilot's interaction exercise below.

## Original pilot generation and verification record

Vertex custom job `4112639277085491200` succeeded on September 13, 2026 using
one Spot NVIDIA A100-SXM4-40GB. All 83 output artifacts were checksum-verified.
Generation itself took 9.04 seconds, excluding provisioning, installation and
checkpoint transfer. The GPU job is finished. The exact model commit/checkpoint,
selection, runtime, source boxes and original dimensions are in `bundle.json`.

Bundle SHA-256:
`8582a85d10a69313fb4672e3580e63f038d01bd6f376f79e7544cb42e0cc2505`.
The [generation audit](outputs/review-pilot-cloud/generation-audit.json) confirms
all 52 RLE masks match their original-grid binary PNGs and mask hashes. Thirty-six
proposals extend outside their boxes; those pixels are retained and flagged for
review, not clipped or treated as an automatic error.

Validation: 168 Python ML tests and 12 JavaScript review-core tests pass.
Browser exercise over localhost verified editing/undo, explicit decisions,
rejection-note gating, reviewer locking, JSON export/resume, and rejecting a
mismatched bundle without replacing existing work. An actual browser export
passed Python validation/import and supplied no anatomy masks under technical
scope. Synthetic fixtures were used for decisions; the delivered pilot has
zero reviews. Direct `file://` navigation is blocked by the browser automation
security policy, so double-click opening was not exercised automatically.
