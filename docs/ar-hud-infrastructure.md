# AR/HUD infrastructure handoff

Current scope: reusable infrastructure. Real surgical assets, Person 3's learning
app wiring, hardware calibration, device acceptance, and actual backup recordings
are deferred to the integration pass at the project owner's request. This is not
a claim that the complete P0 demo has been verified.

## Entry points

| Module | What it supplies |
| --- | --- |
| `apps/web/src/overlays/index.ts` | Public video HUD, result parsing, controls, shared vocabulary and event types |
| `apps/web/src/overlays/VideoHud.tsx` | Video-frame capture, exact result selection, composed canvas, source/warning display, immediate visibility changes |
| `apps/web/src/overlays/useHudControls.ts` | Playback, seek, modes, source/input switching, frame invalidation and selection callback boundary |
| `apps/web/src/camera/index.ts` | Public model HUD, registration parser, readiness panel and recording component |
| `apps/web/src/camera/modelRegistration.ts` | Marker-relative model configuration validation and calibrated 3D projection |
| `apps/web/src/camera/HudDemo.tsx` | Local file integration harness, accessible through the existing Camera prototype tab |

Start from the repository root with `npm run dev`. The integration harness accepts
local video, result JSON, and model registration JSON. It does not upload them.
The original synthetic lesson and Person 3's answer logic are unchanged.

## Person 1 supplies video results

Use the existing FrameResult 1.0.0 schema unchanged. The complete field/color
reference is in [the steps 1–3 contract handoff](ar-hud-steps-1-3.md).
`parseResultSet(rawJson)` accepts a single FrameResult or an array. It rejects
mixed media/dimensions, duplicate source/time records, conflicting frame IDs, and
invalid geometry/provenance. Combine reviewed, ML, propagated, and synthetic
records in the same array only when they actually belong to the same media.

Supply the actual media ID alongside the URL. Original frame timestamps must be
presentation times in milliseconds from media start. The renderer uses
requestVideoFrameCallback's mediaTime and accepts at most 0.5 ms of serialization
rounding. Ambiguous matches are rejected. It never chooses the nearest sparse
annotation outside that tolerance. An unannotated frame has no anatomy overlay.

Review and prediction provenance stay separate. Prediction confidence thresholds
must be supplied by Person 1; an absent/invalid threshold withholds predictions.
Missing/error/unsupported results clear geometry and show Unable to assess.
Successful empty results remain successful. Partial views have dashed outlines.
The current format is a simple polygon contour, not an RLE/binary mask with holes.

## Person 3 connects the controls

```tsx
import { VideoHud, parseResultSet, useHudControls } from "./overlays";

// Create one control hook in the learning app's parent.
const hud = useHudControls({
  frameClock: "external", // VideoHud reports displayed frames; timeupdate is not the clock.
  onDisplayedFrame: frame => { /* update frontend frame state; null clears it */ },
  onAnatomySelection: selection => { /* handle selection; no score is supplied */ },
});
// Set hud.setMedia(media.id) when loading a clip, not during render.
// Parse once when loading results, not once per video frame.
const results = parseResultSet(person1Json);

<VideoHud
  key={media.id + media.src}
  src={media.src}
  mediaId={media.id}
  results={results}
  source={hud.state.source}
  mode={hud.state.mode}
  visible={hud.state.overlaysRequested}
  minimumConfidence={person1Threshold}
  bindVideo={hud.bindVideo}
  onDisplayedFrame={frame => {
    if (frame) hud.reportDisplayedFrame(frame, hud.state.revision);
    else hud.invalidate();
  }}
  onCanvas={setRecordingCanvas}
/>
```

The snippet shows attachment points; variables come from Person 3's app. Use
`hud.play()` (catch its Promise rejection), `pause()`, `seek(ms)`, `showOverlays`,
`setMode`, `setSource`, and `setExperience` for frontend controls. Keep callbacks
current when attaching asynchronous input adapters. A reported frame is a matching
exported frame; null means no verified matching frame is currently available.
Use the default conservative clock policy only for inputs without a frame adapter;
it clears frame identity on generic playback/timeupdate events.

Learn permits annotations. Identify/Assess suppress all answer-revealing labels
and geometry even if visibility is requested. Feedback permits reviewed geometry
or explicitly synthetic fixtures; it cannot use ML as an answer key. The caller
still owns when feedback is entered, response commitment, correctness, timing,
and cumulative hint exposure. The renderer never records an answer.

Video and overlays are copied into the same canvas from each captured frame.
This keeps geometry attached to the displayed image even if the browser misses a
callback. Seek/loading/errors clear the composed view. A source/mode/visibility
change redraws the cached displayed frame immediately. Labels live outside the
image in a dark rail; on narrow screens the rail moves below the image. Fill,
outlines, leader lines, colors, provenance, and warning boxes use the same output
canvas, which is also the recording source. Source-only inputs from a previous
media ID cannot match the new video.

Region selection payloads and stale-selection validation are exposed by
useHudControls. Region hit testing and its attachment to Person 3's identification
interaction remain part of the deferred frontend wiring. Model selections use a
separate registration/anchor identity type rather than video pixel coordinates.

## Physical-model configuration

The implementation uses `js-aruco2` with the ARUCO_MIP_36h12 dictionary, exact-code
matching, and POSIT pose estimation. Supply a configuration to
`parseModelRegistration`, then pass the result to `ModelHud` with the selected
camera stream, learning mode, and visibility. It rejects wrong camera aspect,
invalid pose/depth, duplicate IDs, and invalid anchors. Marker loss removes
anchors immediately; missing camera frames trigger a 250 ms watchdog. It does not
fade or retain a lost pose. The parent must stop camera tracks on exit.

Configuration example for a **synthetic fixture only**:

```json
{
  "modelId": "synthetic-marker-test",
  "provenance": "synthetic_mock",
  "dictionary": "ARUCO_MIP_36h12",
  "markerId": 7,
  "markerSizeMm": 80,
  "calibration": { "width": 640, "height": 480, "fx": 640, "fy": 640, "cx": 320, "cy": 240 },
  "anchors": [
    { "id": "test-center", "structureId": "gallbladder", "positionMm": [0, 0, 0] }
  ]
}
```

For the actual model, replace these example numbers with measured locations and
calibration, and set provenance to `measured_model_locations`. The example's
gallbladder ID is an arbitrary class-mapping fixture, not an anatomical assertion.
The parser validates structure, not whether measurements were actually performed.

Coordinates: millimeters; origin at the marker center; x right and y up in its
canonical upright view. The square's corners are (-s/2,+s/2,0), (+s/2,+s/2,0),
(+s/2,-s/2,0), (-s/2,-s/2,0). Positive z follows the POSIT model axis (away from
the viewer for a front-facing identity pose); verify depth direction during the
calibration pass. Anchor positions include the measured marker-to-model offset.
Use a rigid mount. `markerSizeMm` measures the black square, excluding its white
margin. The harness can save the configured marker as SVG for printing.

Camera calibration is fx/fy/cx/cy at the stated image size. The detector downsamples
to at most 640 pixels wide and scales intrinsics accordingly. Lens distortion is
not modeled; use a rectified input or validate a central, narrow working region.
Small viewpoint changes still require actual-model testing. No device, model,
intrinsics, or anatomical anchor measurements have been assumed to be verified.

## Recording and validation

`HudRecorder` accepts the current composed canvas and `name="video"` or `"model"`.
Start, demonstrate the flow, stop, then save the local recording. It selects a
supported browser format and stops capture tracks during cleanup. The recording
contains the visible HUD and source/warnings, with no audio. Save before switching
input modes. Recording controls are infrastructure; no actual two-mode backup has
been produced in this session.

Run `npm run check` and `npm run build`. Tests exercise result provenance,
timestamp/media matching, missing/malformed output, confidence and mode rules,
letterboxing, selection identities, actual fiducial detection against generated
pixels, pose projection/translation, and invalid calibration. The automated marker
fixture is not hardware validation. Browser visual testing was unavailable because
the in-app browser had no connection.

Integration pass: supply the real clip/results, connect lesson events, select and
calibrate the real camera/model, inspect alignment and bright/dark/blurred/cluttered
readability, run failure recovery, then record both actual modes. P1 smoothing,
class toggles, confidence widgets, CVS panels, and frame-inspection features are
not implemented.

Implementation references: [js-aruco2](https://github.com/damianofalcioni/js-aruco2),
[video-frame callbacks](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback).
