# Person 1 dataset samples → AR/HUD

For the standalone React test page, run **`npm run dev:samples`** and open
**http://127.0.0.1:5174/samples**. It automatically loads the local test folder,
so Person 3's learning app is not needed to begin checking the renderer.
Previous/Next case, overlay controls and learning-mode selection drive the same
renderer API. The displayed-frame and last-click panels expose the frontend
callbacks. Anatomy names in the click panel are hidden whenever overlays are
hidden, text labels are switched off, or the mode is Identify, Assess or
Feedback. Clicks are not scored.

The tester focuses on overlays for the supplied images. The temporary ML-result
example and its separate tab have been removed. Its old query-string link now
opens the same image-overlay tester.

**Overlay appearance** controls fill opacity (0–100%), boundaries, and labels
with pointers independently. **Boundaries only** removes the fill and labels
for alignment inspection. **Reset overlay** returns to the exact mask at 28%
fill, with boundaries and labels visible. **Show overlays** is the master
switch; Identify and Assess always withhold all anatomical layers regardless
of these settings. Appearance controls are disabled outside Learn.

Raster contours use a two-pixel band inside annotated foreground pixels and
a dark contrast halo when drawn. Native mask holes and ignored pixels retain
their transparent fill/contour values. Label anchors are ordered by vertical
position, and the label rail reserves its space when labels are hidden so
the image does not shift or zoom when controls change.

**Reload local samples** re-reads the folder. Automatic loading is supplied by
opt-in Vite development middleware bound to loopback; it exposes only label/index
metadata and the image/mask bytes referenced by supplied hashes. The browser
still validates the imported sample envelopes, hashes and decoded mask contents.
The endpoint is absent from ordinary dev mode, preview and production builds.

Alternatively, start `npm run dev`, open **Dataset samples**, and choose
`apps/web/tests/samples_tst` with **Open sample folder**. The second file picker
accepts all files together when folder selection is unavailable. Imports remain
local and work in the production build as well. Use localhost or HTTPS for
SHA-256 validation. No surgical media is bundled or copied into public assets.

The viewer supports case switching, exact colored masks, boundary outlines,
labels outside the anatomy, approximate polygon preview, instant show/hide,
and the shared Learn / Identify / Assess / Feedback mode vocabulary.

## What the supplied pack contains

The September 13 handoff has several mismatched filenames: `anatomy.json` is
actually the sample README, `verification.json` is a PNG, and some `.png` files
contain JSON or JPEG data. Originals are preserved. The importer identifies
content from bytes and uses each label record's SHA-256 values to pair its
original JPEG and encoded index mask. Folder flattening and renamed files do
not change that pairing. Duplicate conflicting records are withheld.

Two complete cases are available: **116_35325** and **119_77250**, both 854 × 480.
The pack index also lists **4_36700**, but its label envelope is absent. The
viewer reports that case as unavailable rather than guessing a mask pairing.
Replacing the folder selection with a complete pack can make it available.

Each case is an independent still with `frameNumber: 0`, `timestampMs: 0`, and
its own `mediaId`. Original video/frame identifiers remain in
`annotationSource`; they do not establish playable-video timing. This folder
contains neither the MP4 nor the prediction exports referenced in
[Person 1's video handoff](../ml/DEMO_MEDIA.md).

## Input and provenance

`dataset_annotation_sample` is a separate envelope, parsed by
`parseDatasetSample`. Its source remains `supplied_dataset_annotation`.
The existing `FrameResult` schema and source enum are unchanged. Do not pass
the envelope through `parseFrameResult`, or cast it to reviewed/model output.

The default overlay comes from the authoritative `labels-index.png` bytes.
The renderer validates original dimensions, grayscale class indices, the
class mapping, per-class pixel counts and ignored-pixel counts before showing
the image and overlay together. Background 0 and ignored 255 remain transparent;
holes and regions missing from polygon exports are preserved in the raster.
Labels attach to actual foreground pixels, including classes absent from the
polygon approximation. Approximate polygon mode carries an omission warning.
`partial` in these exports means image-border contact only.

Dataset labels have no model confidence or reviewed answer key. Identify and
Assess hide all masks, outlines, leader lines and labels. Feedback withholds
dataset overlays and reports that independently reviewed feedback is missing.
The lesson controller continues to own correctness and response recording.

The UI retains the supplied CAMMA / Endoscapes credit and CC BY-NC-SA 4.0 link.
Keep the original license, source README and provenance files with the local
pack. `samples_tst/` is ignored by Git; ordinary builds contain only the
allowlisted synthetic assets.

## Person 3 component handoff

```tsx
import { DatasetSampleHud, importDatasetSamples } from "./overlays";

// Run on file selection; store the returned pack in parent state.
const pack = await importDatasetSamples(Array.from(files));
// Show pack.issues alongside the picker. Clear the previous selection on reload.

<DatasetSampleHud
  sample={selectedSample} // LoadedDatasetSample or null
  mode={learningMode}    // learn | identify | assess | feedback
  visible={showOverlays}
  view="raster"         // raster | polygons
  appearance={{ fillOpacity: 0.28, showBoundaries: true, showLabels: true }}
  onDisplayedFrame={setDisplayedFrame}
  onSelection={selection => { /* pass to lesson controller; do not echo answers */ }}
  onCanvas={setRecordingCanvas}
/>
```

`onDisplayedFrame` reports the original still identity only after image/mask
validation and rendering, and reports null during replacement and unmount.
Changing samples clears the prior display; late image decodes cannot replace
the current sample. The component owns bitmap cleanup and responds to resize.

The optional selection callback uses the exact index mask even when labels
are hidden. It returns the source, frame identity, original-pixel point,
`structureId` or null, and `annotated | background | ignored`. It never displays
the answer or emits correctness. Clicks on the label rail or letterboxing are
ignored. This callback has its own dataset provenance type; do not feed it
into the video `ResultSource` or relabel it for `useHudControls`.

`appearance` uses the exported `HudAppearance` type. It changes presentation
only; original pixels, masks, provenance and selection coordinates are retained.
Omitting it uses the default mask styling. Opacity is clamped to [0, 1], with
non-finite values falling back to the default. These settings cannot bypass
the component's visibility and learning-mode gates.

For MP4 predictions, use the existing **AR / HUD demo → Video** importer with
Person 1's actual clip, its `FrameResult[]` JSON, declared media ID and threshold.
The dataset stills cannot be used as that clip's annotation timebase.

## Validation and remaining checks

`npm run check` includes synthetic boundary/raster tests and, when the local
`samples_tst` folder exists, a real-pack hash/identity import check. CI skips
only that local-media check when the private/local files are absent.

Local verification also decoded both actual index PNGs with Pillow and checked
them through the raster adapter: six labeled classes per case, all supplied
pixel counts matched, and 9,085 ignored pixels were preserved in case 116.
All six label anchors in each case fall on their corresponding raster class.

The standalone dev page and automatic import were exercised over HTTP, including
both usable cases, the missing-case notice, rejection of foreign origins and
unknown asset URLs, and Vite transformation of the React entry and renderer.
Browser interaction/visual checks remain pending because no browser was
connected in this session. Check actual browser mask decoding, case switching,
resizing, show/hide, Identify/Assess concealment and feedback withholding on
the demo device. Reviewed lesson answers, the third label envelope, video
assets/predictions and physical-model calibration remain separate inputs.
