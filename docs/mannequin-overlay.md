# Camera image and mannequin overlay

Open `/mannequin` on the running React app. With `npm run dev:samples`, the
address is http://127.0.0.1:5174/mannequin. The same setup is also available in
**AR / HUD demo → Training image AR** in the full prototype.

The default is **Camera identification**. See [camera, video and stream setup](live-feed.md)
for preview and the built-in trained-model connection. **Upload video** opens a
local clip; **Stream link** connects direct HTTPS video or HLS media. Select **Training image AR**
to use the camera/marker setup documented below.

Within that view, **Labeled image** uses a training surgical JPEG and its matching
dataset mask. **Camera screen** needs no marker and stays fixed on screen;
**Table marker** places the same image on a flat surface beyond a marker.
See [the image overlay guide](camera-image-overlay.md) for sample loading,
image views, provenance and frontend integration.

## Check the camera and marker

### Open it on a phone

Kevin's cloud-connected test site is **https://holospex-mu.vercel.app/mannequin**,
in Vercel project `kevhuang-3216s-projects/holospex`. Its model runs on Cloud Run;
see [cloud hosting and redeployment](../ml/CLOUD_INFERENCE.md). The older
`holospex.vercel.app` remains in the other account and was not changed.

The phone needs an HTTPS URL. `localhost` and `127.0.0.1` on a phone refer to
the phone itself, so the laptop's development URL will not work there.

Deploy the prepared app and API functions to the linked Vercel project. From the
repository root:

```powershell
npm run build
npm run prepare:phone
vercel login
vercel deploy --cwd runs/holospex-mannequin-phone --yes --prod
```

Open the resulting HTTPS URL with `/mannequin` on the end directly in
Safari on iPhone or Chrome on Android, tap **Start camera**, and allow access.
The rear camera is preferred by default. Live preview needs no reference file;
live anatomy requires the configured trained checkpoint runner.
For a remote source, choose **Stream link** and **Connect stream**; its host
must permit cross-origin playback and frame capture. See
[stream formats and controls](live-feed.md#identify-a-stream-link). Phone and
stream-device compatibility still need testing with the intended source.

For reference-image AR, [prepare a train-split batch](live-feed.md#training-image-references),
then run `npm run prepare:camera-samples` on the laptop. Transfer one generated
`.holospex.json` from `runs/camera-samples/` to the phone's Files app. Choose
**Training image AR**, open the file under **Open camera sample or sample
files**, and press **Start camera**. **Labeled image** and **Camera screen**
provide a test without a marker. A preview appears before camera capture.
Once deployed and the reference transferred, this local image mode does not
need the laptop. Training references and a live inference service are separate inputs.
If the project has Vercel deployment protection enabled, sign in on the phone
as well. Rebuild, prepare and deploy again when changing the app.

The prepared upload contains built browser assets, approved synthetic fixtures
and the mannequin cutout, plus separate placement and identification APIs. It excludes local
training data, weights and portable camera samples. Reference images, marker
processing and model configurations stay in the browser. Live camera uploads
occur after **Start camera** when the model is ready and overlays are visible;
the placement API never receives camera frames. The private `/__local-training/` development endpoint is absent from
the deployed build, and there is no automatic test-sample fallback.
`npm run preview:phone` is available for checking the build locally on port 4174.

References: [Vercel CLI deployment](https://vercel.com/docs/cli/deploy),
[camera secure-context requirement](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

### Run the tracking check

1. Choose **Training image AR**, load a training reference under **Labeled image**,
   and select **Table marker**. Download
   the printable marker SVG. Print at
   100% scale and measure the black square: the default is 80 mm. The complete
   SVG is 100 mm including its white border. Keep that border intact.
2. Place the paper flat on a table, with the marker's top pointing away from
   you. Keep only one copy of the chosen marker in view.
3. Select a camera, press **Start camera**, and allow camera access. More camera
   names may become available after permission is granted.
4. The labeled surgical image appears beyond the marker's top edge. Move back
   enough to see the marker and image, and adjust **Image width on table** to
   fit. The image and labels share the same projection. Placement uses
   approximate camera calibration; the image retains its dataset provenance.
5. Move the camera slightly, hide/reveal the overlays, and cover the marker.
   Losing the marker removes the image and labels. Uncovering it should
   reacquire them. Identify, Assess and Feedback withhold dataset images and
   labels even when Show overlays is checked; only Learn reveals them.

**Marker test** still provides three generic points for isolating tracking
problems. **Mannequin configuration** imports known locations for a physical
model. The earlier procedural surgical renderer remains a separate internal
adapter; it is no longer offered by this setup UI.

Use desktop localhost or an HTTPS deployment on the demo phone. Plain HTTP
over a LAN generally cannot request camera access. Camera processing is local;
there is no ML inference or anatomy recognition in this path.

## Use the mannequin's locations

Select **Mannequin configuration** and import a model registration JSON file.
Loading malformed data clears the old model and reports the error.
Configuration imports do not change the shared
video `FrameResult` schema.

The format and coordinate axes are documented in
[Physical-model configuration](ar-hud-infrastructure.md#physical-model-configuration).
Supply the marker ID, measured black-square size, actual camera intrinsics,
and predefined anatomical anchor positions measured relative to the marker.
Set provenance to `measured_model_locations` only once those measurements are
supplied. Schema validation cannot verify that measurements are accurate.
The marker test uses an explicitly synthetic fixture. Synthetic files show
generic test-point names; measured files use the shared anatomical names/colors.

The marker must remain fixed relative to the mannequin. Loading a new
configuration clears prior anchors immediately. A wrong camera aspect ratio,
missing marker, duplicate visible marker ID, invalid pose, interrupted camera,
or a stall after tracking removes labels and displays a distinct warning.
Calibration is never automatically rewritten for an imported model.

## Frontend connection

`MannequinDemo` is the reusable camera/setup host:

```tsx
import { MannequinDemo } from "./camera";

<MannequinDemo mode={lessonMode} visible={overlaysRequested} />
```

The learning frontend owns mode changes and answer recording. For a custom
setup UI, reuse `ModelHud` with `stream`, `registration`, `mode`, `visible`,
and `interrupted`, plus optional `imageOverlay` for a loaded camera image.
See [the adapter example](camera-image-overlay.md#frontend-adapter).
Optional `onCameraSize` reports native frame dimensions;
`onTracking` reports whether a valid marker pose is available. A valid pose
can still place all configured anchors outside the visible image. `onCanvas`
exposes the composed view to `HudRecorder`. Screen placement does not report
marker tracking. Dataset image feedback is withheld until reviewed answers
are supplied; the learning frontend cannot treat dataset labels as scoring.

## Validation and limitations

Automated checks cover decoding the exported printable SVG with the actual
marker detector, projecting off-center 3D anchors under small synthetic
viewpoint changes, detector downsampling, and rejecting invalid configuration.
Image tests cover sample hashes and provenance, cutouts, matching image/label
projection, and answer hiding. Camera interruption clears both image placements;
table placement also clears on marker loss or invalid pose.
The display keeps its camera rectangle stable when labels are hidden or lost.
Preview capture is capped at 1280 pixels wide; detection is capped at 640.

A single planar marker can have ambiguous poses, especially when distant or
nearly frontal. Lens distortion and mannequin occlusion are not modeled.
The image overlay is a flat texture, and the imported physical-model mode uses
predefined point labels. Neither reconstructs anatomy, recognizes live anatomy
or automatically detects the table. Table image placement requires WebGL 2;
use Camera screen if it is unavailable. Validate the chosen central
working region and actual camera/model measurements before demonstrating
anatomical alignment.

On the actual device, test a clean launch, camera permission denial/retry,
camera switching and disconnection, marker loss/reacquisition, resizing,
lighting, and small viewpoint changes. Use **Record backup**, demonstrate the
flow including tracking loss, stop, and save the recording. Hardware alignment,
browser visual acceptance, and an actual backup recording remain pending.
