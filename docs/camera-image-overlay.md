# Labeled surgical image over the camera

`/mannequin` defaults to **Labeled image**. The original surgical JPEG provides
tissue detail, and its matching index mask supplies colored boundaries and
label positions. This is virtual placement of an existing image, not a 3D
anatomical reconstruction. The source remains **Supplied dataset annotation**;
it is not relabeled as synthetic anatomy, reviewed feedback or ML inference.

## Load a sample

On the laptop, run `npm run dev:samples` and open
http://127.0.0.1:5174/mannequin. The usable cases in
`apps/web/tests/samples_tst` load automatically. **Reload local samples** rereads
the folder. Manual import accepts the folder or a matching label record,
original JPEG and index PNG together; pairing uses hashes, not filenames.

For the [phone demo](https://holospex-mannequin-phone.vercel.app/mannequin), run:

```sh
npm run prepare:camera-samples
```

This validates the local pack and writes portable files, currently:

- `runs/camera-samples/case-116_35325.holospex.json`
- `runs/camera-samples/case-119_77250.holospex.json`

Case `4_36700` is incomplete and is withheld. Transfer one generated file to
the phone's Files app, then select it under **Open camera sample or sample
files**. Each file contains the original image, exact mask and unchanged label
record. Import checks hashes, class mapping, dimensions and provenance;
decoding also checks mask values and counts. Invalid or incomplete samples
do not leave the previous image visible.

Index masks are decoded directly from their PNG bytes, including scanline
filters, without passing through browser image decoding or canvas readback.
This preserves class IDs when browser color handling or privacy protections
would alter the pixel values. The decoder supports the supplied non-interlaced,
8-bit grayscale PNG format and requires browser `DecompressionStream` support.
Existing portable sample files work unchanged.

Portable files are limited to 20 MB and read locally. `runs/` is ignored by
Git and excluded from the prepared deployment. Dataset assets are not hosted
on the public site. Attribution remains Endoscapes-Seg50 / CAMMA and the
Endoscapes authors, with the supplied CC BY-NC-SA 4.0 notice.

## Display and placement

**Full surgical image** preserves the entire still with subtle mask tint and
colored contours. **Anatomy cutout** keeps only annotated anatomy; background,
tools and ignored pixels become transparent. Label anchors come from the
exact mask, including regions omitted from approximate polygon exports.

**Camera screen**, the default, centers the image at its original aspect ratio.
Adjust **Image size** and **Image opacity**. It stays fixed on the screen as
the phone moves and requires no marker. A preview is available before starting
the camera.

Choose **Scene → Mannequin + sample** to place the same surgical image on the
supplied `mannequin.png`. It starts with a smaller illustrative abdominal
placement. Select **Fit anatomy with AI** to request an organ-aware position
and scale, then adjust **Left / right**, **Up / down**, and **Sample width** as
needed. **Reset placement** restores the manual starting position.
Both full-image and anatomy-cutout views work; the photograph, mask boundaries
and anchor positions share one aspect-preserving transform. The
mannequin uses a prepared transparent cutout, so the camera shows through around
the body. The cutout retains the original image dimensions and pixel positions;
sample placement and label registration use the same coordinates. Its source
record is in
[`assets/demo/README.md`](../assets/demo/README.md).

Choosing the mannequin switches to **Anatomy cutout** and **Table marker**. The mannequin,
sample and labels follow the same marker pose; **Mannequin width on table**
controls the complete composite's physical display width. **Camera screen**
remains available for a fixed preview. This is a flat image composite with
AI-suggested or manually positioned anatomy, not measured registration to a physical mannequin
or an anatomical reconstruction. The surgical sample retains its dataset
provenance and still stays local to the browser. The optional
[placement API](anatomy-placement-api.md) receives class names and mask
measurements only; it never receives surgical image bytes or camera frames.

**Table marker** projects the same texture and labels onto a flat surface
beyond the printed marker's top edge. Print the marker at 100%, enter its
measured black-square width, and keep it visible. Adjust **Image width on
table** to set the virtual surface width. This mode uses WebGL 2 and approximate
camera calibration; see [camera and marker setup](mannequin-overlay.md).

Show/hide takes effect immediately. Identify and Assess withhold the entire
image and its labels to prevent answer exposure. Feedback also withholds
them and reports that reviewed feedback is unavailable. Learn displays the
supplied annotations. Camera interruption clears overlays in either placement;
table placement also clears on marker loss or invalid pose.

## Frontend adapter

`MannequinDemo` handles file selection and camera setup. For a custom host,
prepare a validated `LoadedDatasetSample` with `loadImageOverlayAsset`, then
pass an `imageOverlay` to `ModelHud`. Screen placement allows a null
registration; table placement requires the marker registration:

```tsx
<ModelHud
  stream={stream}
  registration={null}
  visible={overlaysRequested}
  mode={lessonMode}
  interrupted={cameraInterrupted}
  imageOverlay={{
    asset,
    view: "scene", // "anatomy" for the mask cutout.
    placement: "screen", // "table" for marker placement.
    opacity: 1,
    widthFraction: 0.8,
    planeWidthMm: 240,
  }}
/>
```

`CameraImageOverlay` is exported from `camera/ModelHud.tsx`.
`loadImageOverlayAsset` and `ImageOverlayAsset` live in
`camera/imageOverlayAsset.ts`. Dispose the asset after detaching it from the
renderer. The transfer helpers `exportPortableSample` and
`importPortableSample` live in `overlays/portableSample.ts`; their envelope is
separate from the shared ML contracts. The frontend still owns lesson logic
and response recording.

For mannequin composition, `composeMannequinAsset` in
`camera/mannequinComposite.ts` returns the same `ImageOverlayAsset` interface.
Its dimensions and anchors are mannequin-image pixels. Dispose it separately
from the input sample, which remains caller-owned. `useMannequinComposite`
loads the bundled reference and withholds an obsolete composite when the
sample or placement changes. Source-mask decoding is unchanged.

Automated tests check byte integrity, provenance, cutouts, shared image/label
projection and mode gating. Actual phone rendering, marker alignment and
recorded backup verification remain device checks; no browser visual
acceptance is claimed. The overlay does not recognize anatomy in the live
camera or determine clinical assessments.
