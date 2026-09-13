# AR/HUD handoff: steps 1–3

This records the first control/contract slice. The current reusable renderer,
marker modules, local harness, and deferred teammate wiring are documented in the
[infrastructure handoff](ar-hud-infrastructure.md). The Camera prototype tab now
opens that harness; DeviceSetup remains a separately exported readiness component.

React + TypeScript + Vite is confirmed by the project owner. Build and integrate
today; test and revise tomorrow. This handoff establishes the control and data
boundaries. Teammate acknowledgment and actual hardware checks remain pending.

## Run the readiness screen

From the repository root:

```sh
npm ci
npm run dev
```

Open the displayed URL, then **Camera prototype**. This existing app-shell entry
now opens the device setup screen; the lesson screen remains separate.

1. Enter the actual demo device/browser and physical model/marker. Confirm the
   availability checkbox only when those items are physically available.
2. In **Video**, choose a permitted local test clip. Press Play, Pause, then Seek
   with a millisecond value. Playback is marked verified only after the media clock
   advances during playback. Selecting a different file resets that result.
3. In **Physical model**, press Start camera. Permission is requested only here,
   with video and no microphone. A playing preview with nonzero dimensions is the
   camera check; API availability alone does not pass it. Stop camera releases it.
4. Switch between inputs. Video pauses when switching; camera tracks stop when
   leaving the model input or setup screen. Disconnecting a camera requires restart;
   temporary camera muting is shown as interruption.
5. Exercise Learn, Identify, Assess, Feedback, show/hide, and all four source choices.
   The policy readout reflects the selected controls; this setup does not render
   anatomy or fabricate frame IDs for arbitrary test clips.

The local clip and device details stay in the browser for this session. No upload,
recording, persistent hardware inventory, or model registration is performed.
Phone camera access requires an agreed HTTPS URL; desktop localhost can be used
on the device running the development server. Test the actual clip format and
browser, not only a codec support declaration.

## Person 1: ML-to-HUD contract

The existing schema **1.0.0** is the implementation baseline, unchanged:
[`frame-result.schema.json`](../contracts/schemas/frame-result.schema.json).
Confirm this baseline with Person 1 before producing new assets. The HUD consumer
uses the shared `parseFrameResult` validator; no second wire schema was created.

| Field | Agreed implementation meaning |
| --- | --- |
| schemaVersion | `1.0.0` |
| mediaId + frameNumber | Media identity plus original zero-based frame number; not an index into a sampled result array |
| timestampMs | Original frame presentation time in milliseconds from media start; fractional milliseconds retained |
| width, height | Original image dimensions, before ML resize/crop |
| coordinateSpace | `original_pixels`; origin top left, x right, y down; image-edge bounds include width and height |
| structures[].structureId | Shared string class ID from anatomy.json; Person 1 maps any dataset integer labels explicitly |
| structures[].instanceId | Unique instance ID within this result |
| structures[].polygon | One simple contour of at least three [x,y] pairs; no binary mask, RLE, holes, or multiple contours in this version |
| structures[].confidence | [0,1], required for predictions; do not invent scores for reviewed geometry |
| structures[].visibility | `visible` or `partial`; unsupported/unseen structures are omitted, not drawn from an earlier frame |
| source | `synthetic_mock`, `reviewed_annotation`, `ml_prediction`, or `propagated_prediction` |
| model | id/version required for predicted and propagated sources |
| propagatedFromTimestampMs | Required for propagation; must precede the target timestamp |
| status | `ok`, `missing`, `unsupported`, or `error` |
| statusReason | Required for every non-ok result, with empty structures |

Shared colors remain sourced from [`anatomy.json`](../contracts/anatomy.json):

| ID | Label | Color |
| --- | --- | --- |
| gallbladder | Gallbladder | #45d6aa |
| cystic_duct | Cystic duct | #f2c94c |
| cystic_artery | Cystic artery | #f08080 |
| cystic_plate | Cystic plate | #ac9cff |
| hepatocystic_triangle_dissection | Hepatocystic triangle dissection | #6cc9ee |
| tool | Surgical tool | #cbd5e1 |

`readHudFrame(value, displayedFrame, selectedSource)` accepts unknown JSON and
returns either `{status: "ready", result, message: null}` or a status/message with
`result: null`. Missing/null input, malformed records, identity/source mismatch,
and unavailable predictions clear the usable result. Availability messages start
with **Unable to assess**. Successful output with zero detections stays `ready`;
it does not establish anatomical absence or a lesson answer.

Still needed from Person 1: contract acknowledgment; permitted clip and original
frame index; clear and uncertain examples; reviewed annotations and provenance;
model identity, supported-class mapping, and the prediction confidence threshold.
Threshold filtering/rendering belongs to the later prediction integration step.
The existing fixture remains synthetic and is not relabeled reviewed.

## Person 3: React controls and callbacks

Public import: `apps/web/src/overlays/index.ts`. Create one `useHudControls()`
instance in the parent that owns the input and pass its values to child controls
and renderers. Creating separate hook instances creates separate state.

| Control | Behavior |
| --- | --- |
| bindVideo | React callback ref for the HTML video element |
| setMedia(mediaId or null) | Set original media identity and clear the displayed frame |
| play() | Promise; invokes video.play and surfaces browser playback rejection to the caller |
| pause() | Pauses the bound video |
| seek(timestampMs) | Accepts finite nonnegative milliseconds; converts to seconds and clamps to loaded duration; throws for unavailable/unseekable video |
| showOverlays(boolean) | Instant requested visibility state |
| setMode(mode) | `learn`, `identify`, `assess`, `feedback`; Identify/Assess are the Quiz submodes |
| setSource(source) | Select one of the four provenance sources; clears previous frame identity |
| setExperience(input) | `video` or `model`; pauses video and clears frame identity; parent owns camera lifecycle |
| invalidate() | Clear frame identity and invalidate pending work |
| reportDisplayedFrame(frame, revision) | Accept original-media identity from the future display adapter; reject old revisions and other media |
| reportAnatomySelection(selection) | Forward a current video region selection to the frontend; reject stale/out-of-bounds context |

`overlaysVisible` is the policy result to pass to a renderer. Identify and Assess
hide answer-revealing geometry regardless of the show request. Feedback allows
reviewed annotations or explicitly synthetic fixtures; ML and propagated output
cannot supply feedback geometry. Model feedback will use known model locations,
with a separate tracking gate supplied by the future registration adapter.

```tsx
import { useEffect } from "react";
import { useHudControls, readHudFrame } from "./overlays";

// Inside Person 3's parent component:
const hud = useHudControls({
  onDisplayedFrame: frame => { /* frame or null -> frontend state */ },
  onAnatomySelection: selection => { /* selected region -> lesson controller */ },
});
const { setMedia } = hud;
useEffect(() => { setMedia(media.id); }, [media.id, setMedia]);
const input = readHudFrame(rawResult, hud.state.displayedFrame, hud.state.source);
// <video ref={hud.bindVideo} src={media.src} playsInline />
// Renderer receives input.result and hud.overlaysVisible.
// Display sourceLabels[hud.state.source] and input.message separately from anatomy.
```

`DisplayedFrame` returns mediaId, frameNumber, timestampMs, width, height. Native
timeupdate events are not an original-frame index. The later video adapter must
resolve presentation time against the supplied media index, capture
`hud.state.revision` before asynchronous work, and report only the matching frame.
Seek, loading/error, playback updates, source/mode/input changes clear identity.
Clear the frontend frame on unmount as well. This step intentionally leaves frame
identity null in the local clip readiness test.

Video selections contain original-pixel `point`, nullable structureId/instanceId,
the displayed frame, source, and revision. No label is displayed and no score is
computed by the callback. Region hit testing arrives with overlay rendering.
The separate `ModelAnatomySelection` type uses registrationId/anchorId/structureId;
its emitter is deferred until there is a tracker that can reject lost registrations.
Never put model/world coordinates into FrameResult.

Person 3 owns lesson progression, response commitment, correctness, response time,
cumulative hint exposure, and response recording. The controls do not advance a
lesson, score a click, or convert a technical failure into a learner answer.

## Verification and remaining gates

Run `npm run check` and `npm run build`. Regression tests cover invalid/missing
results, successful empty output, provenance, late frame callbacks, selection
identity, mode visibility, and seek units/bounds. The build produces apps/web/dist.

Two pre-existing scaffold blockers were repaired: the imported demo loader was
missing (the broad data/ Git ignore hid its folder), and generated contract checks
treated Windows line endings as schema drift. The loader now validates fetched
assets, and the type check normalizes line endings. Schema contents are unchanged.

Browser visual verification was unavailable in the coding session because no
in-app browser connection was available. Actual-device camera permission/preview,
video decoding/playback, physical hardware availability, and model tracking remain
unverified. Run the readiness sequence above on the intended demo device.

Steps 4 onward remain separate: real assets, aligned overlay rendering, synchronized
video, integration with Person 3's lesson, ML uncertainty display, and physical-model
registration. Preserve a complete build and recordings today, then test failures,
alignment, performance, and revisions tomorrow.
