# Web integration guide

This is a runnable architecture scaffold. Its lesson uses a **synthetic illustration**, not surgical footage. The camera mode is a preview, not marker tracking or anatomy detection.

Run from the repository root using the root README commands. Demo source JSON and SVG files live in `assets/demo/` at the repository root; `npm run prepare:demo` validates and copies them into `public/demo/` for the browser. Edit source assets, not the generated public copy. The ML/export owner provides new content through the shared contracts.

## Ownership and boundaries

- **Frontend/learning owner:** `src/lesson/` controls prompts, feedback, and hint exposure. `src/analytics/` stores local attempts and exports JSON. Reviewed answer keys belong to lesson content; model output never grades a question.
- **AR owner:** `src/camera/` is the marker-tracking integration point. `src/input/` owns camera permissions and stream cleanup. A camera frame alone provides no anatomical registration. Draw anchored labels only when tracking is valid, and clear them on tracking loss.
- **ML owner/lead:** produce `FrameResult` files using `@holospex/contracts`. `src/data/` validates these at the boundary. The web app does not train models or interpret inference tensors.
- **Shared display:** `src/overlays/` renders 2D polygons in original-frame pixels. This geometry must never be treated as a 3D marker pose.

## Integrating real video

1. Replace the synthetic image-sequence input with an actual video element and permitted footage. Preserve media IDs and original frame dimensions.
2. Drive checkpoint selection from the playback clock. Export timestamps in **milliseconds** and align predictions to the corresponding media/frame.
3. At a reviewed checkpoint, pause and display only the matching result. The renderer suppresses mismatched media, frame, timestamp, dimensions, and non-OK status. Do not carry old polygons across seek, playback, or missing predictions.
4. Preserve provenance labels. A propagated mask is not a reviewed annotation. Empty detections and failed inference are different states; neither changes a lesson answer key.
5. Author and review clinical questions separately. The bundled synthetic answer checks the scaffold workflow only.

The current lesson labels are hidden initially and must be hidden before answer submission. `hintsVisible` records whether a hint was exposed at any point before submission, even if the learner hides it again. Response timing and answer interaction start only after the checkpoint image loads. Switching modes starts a fresh in-memory lesson attempt; submitted attempts remain in local browser storage. Local storage failures are visible in the UI and never represented as successful saves.

## Camera on a phone

Camera access requires a secure origin: HTTPS on a phone, or localhost on the device running the browser. A plain HTTP LAN URL can show the lesson but usually cannot open a phone camera. The app requests no camera or microphone permission until **Start camera** is clicked. It requests video only, does not upload the stream, and stops tracks when leaving camera mode.

Future glasses integration should replace the input and display adapters. The internal laparoscopic image still comes from a laparoscope feed; an outward-facing glasses camera is a separate input with separate tracking requirements.
