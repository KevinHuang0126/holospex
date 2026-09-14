# ML work plan and commands

For the bounded adaptive Vertex experiment controller, immutable source
packaging, verified result reports, and current optimization options, see
[AUTONOMOUS_TRAINING.md](AUTONOMOUS_TRAINING.md). Its completed comparisons and
candidate decisions are recorded in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

This implements the **Person 1 — ML and imaging pipeline** section of the team
to-do list. The first deliverable is a trained, measurable anatomy segmenter
that exports our shared frame format. CVS answer keys remain independent
reviewed lesson content. Surgical-video segmentation will not recognize a
plastic physical model automatically; the AR teammate owns that camera path.
For the current improved checkpoint and comparison commands, see
[SMALL_ANATOMY.md](SMALL_ANATOMY.md). The commands below also preserve the initial
baseline recipe so the improvements can be reproduced against a control.
The completed loss experiment is documented in [DICE_RESULTS.md](DICE_RESULTS.md):
three matched seeds found that CE + generalized Dice did not improve the
aggregate small-anatomy result. The subsequent [resolution comparison](RESOLUTION_ITERATION.md)
tested 896 × 512 and 1120 × 640; 1120 × 640 increased pooled small-anatomy IoU
and artery recall but also increased false-positive area and inference cost, so
the selected demo checkpoint remains unchanged. Additional data acquisition is
kept separate from these comparisons.
Acquired data, exact cloud object URIs and integration requirements are recorded
in [DATA_CATALOG.json](DATA_CATALOG.json), [DATA_EXPANSION.md](DATA_EXPANSION.md)
and [ADDITIONAL_MASK_DATA.md](ADDITIONAL_MASK_DATA.md). The larger acquisition
does not yet provide a larger fully labeled six-class segmentation test set.

## Data we are using

Use the **official Endoscapes-Seg50** release: 493 segmentation-labeled frames,
split into 343 train / 76 validation / 74 test frames from 30 / 10 / 10 separate
videos. The direct author archive contains extracted frames, not playable
surgical videos. Its full ZIP is about 6.29 GB; our downloader retrieves only
the labeled image/mask pairs and metadata using HTTP ranges.

Sources: [official release](https://github.com/CAMMA-public/Endoscapes),
[benchmark report](https://arxiv.org/abs/2312.12429),
[PhysioNet description](https://physionet.org/content/endoscapes-2023/1.0.0/).

The author release is described as **CC BY-NC-SA 4.0, non-commercial scientific
research**. Retain the downloaded LICENSE/README and source record. This does
not grant an unrestricted commercial-product license. A model checkpoint and
example outputs do not erase the dataset's terms. The alternate PhysioNet route
has its own registered-access/data-use-agreement process; the downloader uses
the separate public URL explicitly linked by the authors.

The class map is read from the actual downloaded file, not guessed from COCO:

| Semantic PNG value | Source label | Internal anatomy ID |
| --- | --- | --- |
| 0 | background | background |
| 1 | cystic_plate | cystic_plate |
| 2 | calot_triangle | hepatocystic_triangle_dissection |
| 3 | cystic_artery | cystic_artery |
| 4 | cystic_duct | cystic_duct |
| 5 | gallbladder | gallbladder |
| 6 | tool | tool |

The model predicts **seven channels**: the six exported structures plus
background. The label for triangle dissection refers to the annotated region;
do not reinterpret it as an independently verified CVS criterion.

The downloaded masks also contain **255**, which the maintainer identifies as
uncertain pixels to exclude from training and evaluation.
[Maintainer explanation](https://github.com/CAMMA-public/Endoscapes/issues/5#issuecomment-2161474399)
Pass this policy explicitly; ignored pixels remain 255 after conversion and
never become background. One validation frame, `153_32700`, contains an
unexplained value **7** and is quarantined without modifying the original.
This leaves **343 train / 75 validation / 74 test** usable images. The manifest
records the exclusion and ignored-pixel counts; other unknown IDs still fail.

## Install and acquire

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e './ml[train]'
.venv/bin/python -m holospex_ml doctor
.venv/bin/python -m holospex_ml download-endoscapes --output-dir ml/data/endoscapes
```

Downloads are explicit; importing or training does not download this dataset.
The downloader validates byte ranges and ZIP CRCs, materializes only selected
files, and records the source URL, archive ETag, and file hashes in the dataset
directory. Re-running verifies existing files and resumes missing ones. It
does not silently accept a server response containing the whole archive.

`--metadata-only` retrieves the class map, annotation JSONs and split lists
without image/mask pairs. It is useful for access/layout checks, but it is not
enough to train. Raw data, weights and run outputs are ignored by Git.

## Prepare and inspect before training

The filename frame-number timebase is an **explicit input**. The direct
archive's annotation names are spaced by 750 frames and documented annotation
spacing is 30 seconds, suggesting 25 fps; PhysioNet separately describes
30-fps preprocessing. Treat these as different pieces of evidence, not an
automatic conversion. Verify against any actual video before integration.
The following uses 25 fps as the archive working assumption; the manifest
records it. Training segmentation does not depend on these timestamps.

```sh
.venv/bin/python -m holospex_ml prepare-endoscapes --data-root ml/data/endoscapes --fps 25 --ignore-source-id 255 --exclude-frame 153_32700 --output ml/outputs/endoscapes-manifest.json
.venv/bin/python -m holospex_ml preview --manifest ml/outputs/endoscapes-manifest.json --split train --limit 6 --output ml/outputs/annotations-train.png
```

Preparation checks image/mask dimensions, actual pixel IDs, class mapping and
video-separated splits. A missing mask is not an all-background training
example. Inspect the original-plus-annotation contact sheet for conversion
mistakes before spending time on optimization. The preview shows supplied
annotations; it is not an ML prediction or a newly reviewed answer key.

## Model choice and first run

Our baseline is **DeepLabV3 with a MobileNetV3-Large backbone** from Torchvision.
It has a small official generic segmentation checkpoint. Its existing labels
are COCO/VOC objects, so we replace both output heads with seven-channel heads
and fine-tune on Endoscapes. Loading generic weights alone does not produce
surgical anatomy recognition.
[Model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html)

The heavier official Endoscapes benchmark framework uses an older
CUDA/OpenMMLab stack. Our choice favors getting a reproducible baseline running
on this Mac and retaining a CUDA path for a cloud GPU.
[Official benchmark implementation](https://github.com/CAMMA-public/SurgLatentGraph)

Keep downloads and checkpoints inside the workspace:

```sh
TORCH_HOME=ml/weights .venv/bin/python -m holospex_ml train --manifest ml/outputs/endoscapes-manifest.json --output-dir ml/outputs/baseline-001 --device mps --epochs 3 --batch-size 2 --width 448 --height 256 --lr 0.0003
```

Use `--device cuda` on an NVIDIA GPU or `--device cpu` when no accelerator is
available. `auto` checks CUDA, then MPS, then CPU. Verify the printed device;
an execution sandbox can hide the Mac's GPU even when it exists.

A new output directory represents a new experiment. The run saves best/last
checkpoints, configuration, history, and validation metrics. Training uses
paired image/mask resizing, nearest-neighbor masks, ImageNet normalization,
and frozen BatchNorm statistics so a singleton batch is valid. Seeds and
versions support repeatability, but do not promise bitwise reproducibility
across accelerator types.

For a fast wiring test, add `--limit-train 8 --limit-val 2 --epochs 1` and use a
separate output directory. Such a limited run is labeled in checkpoint
metadata and is not a performance benchmark. `--no-pretrained` is available
for offline tests; it is not the recommended training initialization.

The first run exposed strong class imbalance: background, gallbladder and tools
occupy about 97% of training pixels. An optional controlled comparison uses
`--class-weighting balanced` with otherwise identical settings. The weights
come only from the selected training masks at model input resolution, using
capped inverse-square-root frequencies; the run configuration saves the exact
counts, formula and weights. Validation/evaluation loss stays unweighted.
Use a separate output directory and choose the checkpoint on validation, never
on the public demo video or test images.

## Evaluate and export

Inspect validation loss and foreground per-class Dice/IoU before choosing
thresholds or training settings. The best checkpoint is selected using
validation foreground macro-IoU. Background is excluded from that selection
score; absent classes are reported explicitly instead of receiving a free
perfect score. Test data is reserved for the final evaluation:

```sh
.venv/bin/python -m holospex_ml evaluate --manifest ml/outputs/endoscapes-manifest.json --checkpoint ml/outputs/baseline-001/best.pt --split test --device mps --output ml/outputs/baseline-001/metrics-test.json
```

Metrics are measured at the configured model input resolution (448 × 256 in
this run), with ignored pixels excluded. They are not an original-resolution
benchmark comparison. Keep the same ignore policy for training and scoring.

When comparing different input resolutions, use the original-grid evaluator on
**every** candidate, including the old baseline:

```sh
.venv/bin/python -m holospex_ml evaluate-original --manifest ml/outputs/endoscapes-manifest.json --checkpoint ml/outputs/baseline-002-balanced/best.pt --split val --device mps --output ml/outputs/baseline-002-balanced/metrics-val-original.json
```

It restores logits to the original annotation dimensions before argmax, leaves
ground truth at native resolution, and excludes source 255. Reports include
per-class precision/recall, the four smaller anatomy classes' mean IoU/Dice,
per-video confusion, and class scores averaged equally across videos. A video
with more labeled frames has more influence on the pooled-pixel score, but
does not receive extra weight in the equal-case score. The report states how
absent classes are handled. These scores are **not directly interchangeable**
with the input-resolution validation numbers saved during training.

The default split for `evaluate-original` is validation. Keep further tuning on
that split; the test split was already inspected for the first baseline.

Generate a comparison across validation cases before handing outputs over:

```sh
.venv/bin/python -m holospex_ml compare --manifest ml/outputs/endoscapes-manifest.json --checkpoint ml/outputs/baseline-001/best.pt --split val --limit 6 --device mps --output ml/outputs/baseline-001/comparison-val.png
```

The sheet shows the original frame, supplied annotations, and unfiltered model
argmax. It selects cases deterministically without ranking prediction quality.
Its sidecar records frame identities and the separately filtered HUD export.

For focused before/after inspection, run `compare-small` with
`--before-checkpoint OLD_PATH --after-checkpoint NEW_PATH --manifest MANIFEST_PATH
--device mps --output COMPARISON.png`. It selects one validation frame per small
class by median annotation area. Both models see the whole frame; only the
display is cropped around the annotation. The displayed scores cover the whole
original frame, so mistakes outside the crop still count. Examples are selected
without consulting model predictions or confidence.

For a selected image, use its actual media/frame identity from the manifest:

```sh
.venv/bin/python -m holospex_ml predict --checkpoint ml/outputs/baseline-001/best.pt --image PATH_TO_IMAGE --media-id VIDEO_ID --frame-number FRAME_NUMBER --timestamp-ms VERIFIED_TIMESTAMP --device mps --threshold 0.5 --output ml/outputs/prediction.json
.venv/bin/python -m holospex_ml validate ml/outputs/prediction.json
```

This exports HUD JSON, a raw argmax label PNG, and a metadata sidecar. The score
is mean uncalibrated softmax over accepted component pixels. A threshold is a
display filter, not a clinical assurance or an automatic `cannot_determine`
answer. Set it using validation behavior; do not lower it just to make the demo
show more labels. An `ok` result with no exported structures is valid.

Logits are resized back to original dimensions before extracting geometry.
The polygon-only HUD cannot represent holes or self-touching boundaries; this
exporter withholds those components and reports the count. The PNG retains the full predicted label
map. The `visibility` field only flags contact with the image boundary; no
occlusion estimator has been trained.

## Video handoff

For an actual playable clip, use decoded presentation timestamps rather than
the Endoscapes filename-fps assumption. The video CLI accepts a prepared,
video-only clip whose first decoded presentation timestamp is zero. See
[DEMO_MEDIA.md](DEMO_MEDIA.md) for the selected public source and preparation
command. It rejects source files requiring an implicit clock offset.

```sh
.venv/bin/python -m holospex_ml predict-video --checkpoint ml/outputs/baseline-001/best.pt --input PATH_TO_VIDEO --media-id UNIQUE_CLIP_ID --device mps --output ml/outputs/demo-video/predictions.json
.venv/bin/python -m holospex_ml validate ml/outputs/demo-video/predictions.json
```

This loads the model once and processes consecutive decoded frames. Frame
numbers are zero-based within this clip. Timestamps are decoded PTS multiplied
by the frame timebase; the sidecar retains the exact PTS and timebase.
Do not assign Endoscapes case/frame identities to unrelated public footage.
The export records source hashes, dimensions, model identity and raw masks.
`--max-frames` explicitly creates a partial wiring check; omit it for the full
clip. Missing or nonmonotonic timestamps and changing dimensions fail instead
of silently guessing an alignment.

Every exported frame is a direct prediction. No temporal propagation or
smoothing is applied. The renderer must use the same clip and clear results
outside their matching frame. A public unlabeled clip supports integration
testing; it does not supply ground truth for an accuracy score or CVS answers.
[PyAV timing documentation](https://pyav.org/docs/stable/api/time.html)

## To-do gates and next steps

1. **Data gate completed:** actual pairs are downloaded, masks inspected,
   classes audited, video splits checked, and timing assumptions recorded.
2. **Baseline gate completed:** two fine-tunes, validation selection, one held-out
   evaluation, and original-coordinate exports are recorded in [STATUS.md](STATUS.md).
3. **Video export completed:** all 120 frames of the prepared public excerpt
   have validated predictions. Connect the [media handoff](DEMO_MEDIA.md) to the
   web player and verify playback; that integration is still outstanding.
4. **Reviewed training targets prepared:** the [completed pilot review](REVIEW_PILOT_RESULTS.md)
   supplies 49 approved masks across 18 additional training images. The lead
   confirmed both flagged edits and selected the [partial-target policy](PARTIAL_TRAINING.md):
   label uncontested reviewed pixels, ignore unknown and conflicting pixels,
   and keep original validation/test samples unchanged. The combined manifest
   has 361 train / 75 validation / 74 test images and passes loader/loss checks.
5. **Reviewed-data comparison completed:** [six paired A100 runs](REVIEWED_DATA_RESULTS.md)
   improved small pooled IoU in all three seeds, averaging 24.40% → 27.23%,
   and equal-case IoU 22.59% → 24.65%. Continue with another bounded review
   batch, concentrating on remaining plate errors, then re-evaluate. The public clip
   is not labeled evaluation data. Clinician review of lesson content remains
   separate from model metrics.

Temporal smoothing, propagation and live inference come after the measurable
offline baseline. No automated CVS classifier is included in this first model.
