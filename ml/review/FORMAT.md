# Candidate mask review handoff format v1

This is an offline ML annotation-review format, separate from FrameResult and
Lesson. A review never supplies a lesson answer. All proposals remain model
outputs until a named person records a decision. No review automatically starts
training. Root coordinates changes to this agreement across generation/UI/import.

## Proposal bundle: bundle.json

Top-level: `formatVersion: "1.0.0"`, `artifactType: "candidate_mask_bundle"`,
`bundleId` (stable nonempty string), `createdAt` (UTC ISO string),
`sourceManifestSha256`, `selection` (documented object), `generator` (provenance
object), `images` (nonempty array), `classes` (array of objects with
`structureId`, `label`, `color`). Pilot: 20 different official TRAIN cases;
include candidates only for duct, artery, plate and triangle source boxes.

Each image: `id` (source filename stem), `videoId` (string), `frameNumber` (int),
`split: "train"`, `width`, `height`, `imagePath` (safe relative path),
`imageSha256` (original file digest), `candidates` (nonempty array).
Each candidate: `id: "endoscapes-box-<annotationId>"`, `sourceAnnotationId` (int),
`structureId`, `bboxXYWH` (source original-image pixel box),
`source: "model_generated"`, `mask` (RLE below), `maskSha256` (SHA-256 of decoded
row-major uint8 0/1 bytes, not PNG), `proposalScore` (finite number or null),
`scoreMeaning` (SAM predicted overlap quality, not anatomy confidence),
`warnings` (array of strings). IDs unique throughout the bundle.

Mask: `{encoding:"rle-row-major-v1",width:854,height:480,counts:[...]}`.
Counts alternate zero/one runs in row-major y-then-x order, beginning with zeros.
An initial zero count is allowed; all subsequent counts must be positive.
Counts sum to width*height; an empty mask is [width*height]. This is not COCO RLE.
Keep holes/disconnected components. Never clip masks to rectangles silently.

Generator records official model/repo/checkpoint identity, exact revision,
checkpoint SHA and software/device/runtime, prediction parameters, and timestamps.
The source image and proposal pixels are immutable. Edited masks belong only to
review decisions. Record outside-box/empty/large-mask flags as review cues;
no model score or geometric flag means an annotation is correct.

## Browser embedding

The packager embeds:
`window.HOLOSPEX_REVIEW = {bundle: <original bundle>, bundleSha256: <SHA256 of
original bundle.json bytes>, images: {<imageId>: <JPEG data URL>}}`.
The renderer template has token `__HOLOSPEX_REVIEW_DATA__` in a script assignment
and no external fonts/scripts/network requests. The packager replaces that token
with safe serialized JSON (escape < so source strings cannot close a script).
The resulting single review.html works offline via a normal browser. Browser
storage is best effort; explicit downloadable JSON is the portable save format.

## Exported review JSON

Top-level: `formatVersion: "1.0.0"`, `artifactType: "candidate_mask_review"`,
`bundleId`, `bundleSha256`, `reviewer: {name: <nonempty>, reviewScope:
"anatomy"|"technical"}`, `exportedAt` (UTC ISO), `decisions` (array).
Every decision: `candidateId`, `imageId`, `imageSha256`, `proposalMaskSha256`,
`decision: "pending"|"accepted"|"edited"|"rejected"|"needs_expert"`,
`mask` (current full RLE), `notes` (string), `reviewedAt` (ISO or null),
`reviewMilliseconds` (nonnegative finite number).
The exported review may be partial; missing/pending entries mean unreviewed.
Accepted pixels must equal the original proposal. Edited pixels must differ;
any stroke/reset after a decision returns that candidate to pending until the
reviewer explicitly decides again. Empty masks cannot be accepted/edited.
Rejected/needs_expert decisions require a note, and no such pixels are eligible.
No technical-scope decision becomes anatomy-approved training supervision.
Reviewer name/scope changes reset decisions or are blocked until a fresh session;
never attribute another person's saved decisions to a new reviewer.

Import/resume must verify bundle and candidate/image/proposal identities,
RLE validity/dimensions, decision semantics and reviewer fields BEFORE mutating
current state. Do not merge reviews from different people silently. Offer an
explicit replace operation for a valid same-bundle review, preserving its author.

## Import back into ML

Validate review JSON against original bundle/file identities. Produce a review
receipt plus separate per-candidate binary PNGs for anatomy-scope accepted/edited
masks only. Keep all other pixels unknown and all other decisions noneligible.
Do not combine conflicting candidate masks into a dense class target silently.
No automatically generated mask, technical-only check, or reviewer name proves
anatomical correctness. All new supervision is partial until coverage is reviewed.

## Derived training targets after a recorded resolution

`prepare_training.py` consumes the original bundle/review, a base Endoscapes
manifest and a separate `review_training_resolution` record. That record binds
the exact input hashes, the lead's clarification evidence, `ignore_conflicts`
and `ignore` policies, and any confirmed edited candidates. It never rewrites
the original review or attributes later lead clarification to the reviewer.

Same-class masks are unioned. Pixels with exactly one eligible class receive its
base-manifest source ID; multi-class conflicts and unreviewed pixels receive
ignore ID 255. The resulting PNGs use the training convention, not the imported
binary-mask convention. Original base samples, validation and test are retained;
only additional TRAIN cases with positive targets are appended. See
[partial-target preparation](../PARTIAL_TRAINING.md) for commands and checks.
