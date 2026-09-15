# Camera, image, video and stream anatomy identification

`/mannequin` and the full prototype's **AR / HUD demo** open **Camera identification**.
Choose a camera or USB video capture device, then press **Start camera**.
The preview stays live on the device. When the trained model is ready, press
**Capture & identify** to freeze one displayed frame and send its JPEG through
the app's built-in `/api/identify` connection. Anatomy appears on that exact
captured image, which stays frozen for inspection after the result returns.
**Retake** clears the capture and returns to the live preview; press
**Capture & identify** again when the next view is ready. Starting the camera,
waiting in preview or retaking does not send frames automatically.
**Stop camera** releases the selected device. **Upload image** previews a local
still and waits for **Identify image** before sending it to the model.
Choose **Upload video**
to select a local clip, or **Stream link** to connect direct HTTPS media. Both
identify sampled frames using the same trained model. The model connection and
confidence setting are supplied by the app; identification has no assessment workflow.

Camera preview works while the model is unavailable. **Refresh model** checks
readiness; failed checks retry automatically after ten seconds, even before
media is selected. The app also checks immediately when the page becomes
visible or the network comes back online while the page is visible. Camera
access needs HTTPS or desktop localhost. The model
expects surgical imagery; an outward-facing camera cannot reveal anatomy
inside a mannequin. **Training image AR** provides the separate reference
visualization on a mannequin or marker.

The completed camera result does not expire after six seconds while its
captured image remains selected. The six-second limit still applies to the
identification request itself. A failed request leaves a frozen frame with
an unavailable message; retake to capture again. Show/hide and appearance
controls preserve the captured image and result without identifying a new
frame. Hiding overlays does not cancel a request already started; it can finish
while anatomy is hidden. Capture is disabled until overlays are shown again.
Replacing or stopping the camera, a camera interruption, a page visibility
change or **Refresh model** clears the capture and cancels pending work.
Camera capture does not change continuous identification for uploaded videos
and stream links.

## Identify an uploaded image

Select **Upload image** and choose a surgical JPEG, PNG or WebP from the device's
file picker. Files are limited to 20 MiB, with decoded dimensions no larger than
16,384 pixels per side and 40 megapixels total. Convert HEIC/HEIF files to JPEG
or PNG before selecting them.
The preview loads locally without camera permission or result JSON. Selecting
the file does not send it to the model. When the model is ready and overlays
are shown, press **Identify image** to send one JPEG through `/api/identify`.
The result stays on the exact image that was sent, with the same source labels,
anatomy colors and display controls used for camera and video identification.
Use **Identify again** to request another result, or **Retry identification**
after an error. Replacing or removing the file immediately clears its image,
labels and pending work.

Decoding honors the image's orientation. The displayed and submitted pixels fit
within 1280 x 720 without upscaling, preserving aspect ratio; labels use those
same resized dimensions. This supports portrait images without stretching the
anatomy. The submitted JPEG is re-encoded from the oriented pixels rather than
passing the source file's EXIF metadata to the runner.

Apply a new **Confidence cutoff (%)** to clear the old result and pending work
while keeping the selected image. Press **Identify image** again to use the new
setting. Showing overlays, changing their appearance or waiting for model
readiness never starts an image request automatically. The six-second request
limit covers encoding, transit and inference; a completed result does not expire
while its image remains selected. No model retraining or new API is needed.

## Identify an uploaded video

Open `/mannequin`, select **Upload video**, and choose a surgical video from
the device's file picker. No camera permission or precomputed result JSON is
needed. The video is decoded locally; when overlays are enabled and the model
is ready, sampled JPEG frames go through `/api/identify`. The full video and
its filename are not sent to the model.

The first loaded frame starts paused. Use **Play video**, **Pause video**, the timeline
or **Replay video** to inspect the clip. Finite videos also offer 0.25x, 0.5x
and 1x playback speed. Model results are drawn over the exact
captured image that produced them, with its timestamp and processing delay.
Seeking either direction clears old predictions and requests the selected
frame. Pausing requests the stopped frame; its matching result can remain
visible while that frame stays selected. **Inspect displayed frame** freezes
the currently displayed image and retains its matching identification, or
identifies that frozen image if no result is available yet. The paused
view does not jump ahead to the decoder's newer playback position. Resume
playback from that inspected position. Replacing/removing the file, switching
input or hiding the page cancels pending identification. Hidden tabs pause
uploaded playback. The recording controls save the composed view.

Uploaded captures fit within 1280 x 720 while preserving aspect ratio. The
model's geometry uses those capture dimensions, so large and portrait clips
scale together with their overlays. Unsupported formats/codecs report a
playback error; MP4 with H.264 is a useful device-testing choice. Browsers
without video-frame callbacks allow preview during playback and identification
while paused. Model inference on this laptop is slower than video playback,
so the identified view updates at the model's processing rate.

In the full prototype, **Video + saved results** remains available for existing
clip/result-JSON workflows. That view is independent of uploaded-video inference.

## Identify a stream link

Select **Stream link**, enter a direct HTTPS media address, choose **Link format**
and press **Connect stream**. **Automatic** recognizes `.m3u8` paths as HLS;
choose **HLS stream (.m3u8)** explicitly when an HLS address has no `.m3u8`
extension. **Direct video (MP4 / WebM)** uses the browser's media decoder.
After editing the address or format, press **Reconnect stream** to apply it.
**Disconnect stream** releases the source and clears its predictions.

YouTube and Vimeo watch pages and RTSP camera addresses are not direct media
inputs. Obtain a browser-compatible HTTPS HLS source or a direct video URL
from the source provider. The browser loads media directly from that host;
the identification server does not proxy or convert the stream. No camera
permission is needed for this mode.

The media host must allow anonymous cross-origin access from the app. For HLS,
that includes GET access to every playlist, segment and other referenced
resource. A link that plays on the provider's own page may still fail these
requirements. See the [HLS.js CORS requirements](https://github.com/video-dev/hls.js#cors)
and [media crossOrigin behavior](https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/crossOrigin).

Use **Play feed** and **Pause feed** for live playback, or **Play video** and
**Pause video** for finite media. Live sources offer **Go live**
when a live position is available; finite video links support the timeline
and **Replay video**. Seeking, reconnecting and source interruptions invalidate old
predictions. Results remain paired with the sampled image that produced them,
and share the uploaded-video capture limits and 6,000 ms inference budget.
**Inspect displayed frame** holds the exact displayed image and matching
result for finite video links and live streams. **Play feed** releases a
frozen live frame and resumes the stream; it does not promise to return to
an expired live/DVR timestamp. **Go live** releases the frozen frame and
seeks to an available live position. Playback-speed controls remain limited
to finite media.
The source URL stays in browser state and is excluded from model requests,
application logs and saved recordings; the model receives sampled JPEG frames
and frame identity only. Recordings contain the composed view.

HLS playback loads HLS.js 1.7.3 on demand where supported, with native HLS
playback as a fallback. Availability still depends on the source codec, host
configuration and browser. Actual stream hardware, phone playback and end-to-end
stream latency require validation with the intended source.

## Start Person 1's identification model

The implementation reuses `holospex_ml.inference.SegmentationAdapter` and its
checkpoint preprocessing, class mapping, original-resolution logits and
polygon export. It loads trained weights once and processes captures in memory.
It does not retrain, download a checkpoint, use OpenAI to identify anatomy,
or retrieve a similar training annotation as a live prediction.

Person 1's September 14 [promoted model](../ml/CURRENT_MODEL.md) is
DeepLabV3–ResNet50, batch-002 seed 42, version
`2026-09-14T04:29:29.757933Z-epoch-34`. The live runner defaults to
`ml/weights/current/best.pt`, verifies the published byte count and SHA-256,
and defaults to Person 1's documented 0.5 confidence cutoff. This replaces the
historical `small-004-resolution` selection. Pulling Git updates the code and
handoff; the trained weights must be retrieved from the private artifact store.

On the model host, install Python 3.11+ and the ML environment described in
[the ML runbook](../ml/TRAINING.md). From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e './ml[train]'
```

On macOS/Linux use `.venv/bin/python` for pip. Install the promoted checkpoint
from a file supplied by Person 1 (or use the authenticated collection command
in [CURRENT_MODEL.md](../ml/CURRENT_MODEL.md)):

```sh
npm run model:prepare -- --checkpoint PATH_TO_DOWNLOADED_BEST_PT
npm run model:serve
```

`model:prepare -- --url HTTPS_DOWNLOAD_LINK` accepts a direct HTTPS download
link. Installation verifies the pinned 168,351,963-byte checkpoint and checksum
before publishing to the ignored stable path, and refuses to replace different
existing weights. With no arguments, `model:prepare` verifies the installed
checkpoint. It never downloads training images. The runtime itself does not
download weights or create random substitutes.

`model:serve` uses the repository's `.venv` on all three platforms and defaults
to port 8765 with automatic CUDA/MPS/CPU selection. To explicitly run another
trained model, pass both `--checkpoint PATH` and `--threshold NUMBER`; custom
models do not inherit the promoted model's cutoff. Readiness returns the loaded
model identity and cutoff. Checkpoint preprocessing, class mapping and
original-resolution coordinates come from Person 1's existing adapter.
The loader accepts the Endoscapes base dataset and its explicitly named
`+reviewed-partial` additions used by the promoted training batches, while
still requiring the same anatomy channel order.
An absent checkpoint or runtime dependency prevents readiness. The cutoff
filters uncalibrated model scores; it does not score lesson answers.

Then run `npm run dev` (or `npm run dev:samples` for local training references).
Vite forwards `/api/identify` to `http://127.0.0.1:8765/identify` by default.
Browser requests stay on the app's own origin; no client-side URL configuration
or cross-origin access to the model is needed.

To test the built frontend locally, run `npm run build`, then
`npm run preview:phone` and open `http://127.0.0.1:4174/mannequin` on the laptop.
Vite preview provides the same identification bridge and loopback default as
development; the trained runner must still be running. Both servers load the
private identification URL/token from the repository's environment files or
their process environment at startup. If the runner requires a token, supply
that same token to the Vite process; it does not read `runner-token.txt`
automatically. Restart dev/preview after changing those settings. Preview uses
Vite's production mode locally; Vercel's environment variables remain separate.

## Phone deployment

`npm run build` followed by `npm run prepare:phone` packages the web app and
separate placement and identification bridge functions. The trained Python
model runs on its own machine; the Vercel function forwards requests to it.
Configure these **server-only** variables on the Vercel project:

| Variable | Value |
| --- | --- |
| `HOLOSPEX_IDENTIFICATION_URL` | Full HTTPS address of the hosted runner's `/identify` route |
| `HOLOSPEX_IDENTIFICATION_TOKEN` | Shared private bearer token, also set on the runner |

The runner defaults to loopback. To expose it behind your HTTPS reverse proxy,
set the token and use `--host 0.0.0.0`. Non-loopback binding requires a token.
The bridge rejects redirects and never accepts a destination supplied by the
browser. Neither endpoint configuration nor tokens enter the browser bundle.
Without the production endpoint, readiness returns HTTP 503 and the app shows
**Identification model connection unavailable** with automatic retries.
A phone's localhost refers to the phone,
so the model needs a reachable host for deployed identification to work.

### Restart the current Windows phone setup

The current phone deployment uses an authenticated runner on the laptop's
loopback interface and a Cloudflare Quick Tunnel. Keep the laptop awake and
online with both processes running for live identification. The deployed
camera preview remains available while the runner or tunnel is stopped.
This tunnel connects to loopback, so the runner does not need `--host 0.0.0.0`.

From the repository root, restart stopped processes in two PowerShell terminals.
In the first, reuse the existing private token and start the installed model:

```powershell
$env:HOLOSPEX_IDENTIFICATION_TOKEN = (Get-Content -LiteralPath 'runs/live-identification/runner-token.txt' -Raw).Trim()
npm.cmd run model:serve
```

In the second, start the existing tunnel executable:

```powershell
.\runs\live-identification\cloudflared.exe tunnel --url http://127.0.0.1:8765 --no-autoupdate
```

The token file is ignored and local; keep its value out of Git and terminal
output. Reusing it preserves the match with the sensitive production
`HOLOSPEX_IDENTIFICATION_TOKEN` variable. A restarted Quick Tunnel can receive
a different HTTPS address. If it changes, use the Vercel project's environment
settings to replace the sensitive
production `HOLOSPEX_IDENTIFICATION_URL` with the new HTTPS address printed
by the tunnel, followed by `/identify`. Leave the existing token unchanged.
Then redeploy:

```powershell
vercel.cmd deploy --cwd runs/holospex-mannequin-phone --yes --prod
```

After deployment, open the phone app and choose **Refresh model**. Changes to
app code still require `npm run build` and `npm run prepare:phone` before
redeployment. The frame budget remains 6,000 ms across capture encoding,
network transit and inference; tunnel availability does not change that bound.

The optional **Fit anatomy with AI** placement feature remains independent.
Its OpenAI key does not configure live identification. Camera mode sends a
single JPEG only after **Capture & identify** is pressed with the model ready.
Camera preview sends no camera images, including while the model is available.
Uploaded images likewise stay local until **Identify image** is pressed; only
the resized JPEG and its frame identity are sent, without the original file's name.
Uploaded videos and stream links continue sampling frames while identification
is enabled. Hiding their overlays pauses identification; stopping or replacing
an input cancels pending frame work. Camera **Retake** also cancels its pending
capture and requires another explicit capture before sending an image.
Hiding camera overlays preserves the capture and any request already started;
hiding the page or interrupting the camera clears it and cancels pending work.

## Frame contract and rendering

`GET /api/identify` returns:

```json
{
  "status": "ready",
  "model": { "id": "checkpoint-model-id", "version": "checkpoint-version" },
  "minimumConfidence": 0.5,
  "supportsMinimumConfidence": true,
  "dataset": "Endoscapes-Seg50"
}
```

Readiness supplies the runner's default cutoff. A true `supportsMinimumConfidence`
enables the app's **Confidence cutoff (%)** input. Enter 0–100 and select
**Apply cutoff**; typing alone does not affect identification. Applying a new
value cancels prior work and clears its result. Camera mode returns to preview
and waits for **Capture & identify**; image mode keeps the selected image and
waits for **Identify image**; video modes identify using the new setting.
Legacy hosts without the capability continue using their default cutoff.
`POST /api/identify` sends JSON with:

| Field | Value |
| --- | --- |
| `frame` | `mediaId`, `frameNumber`, `timestampMs`, `width`, `height` |
| `imageBase64` | Base64 JPEG containing that exact capture, without a data-URL prefix |
| `minimumConfidence` | Optional finite number from 0 to 1, converted from the applied percentage; omission uses the runner default |

`mediaId` identifies a capture session. `frameNumber` is a monotonic capture
counter; skipped inference frames can create gaps. `timestampMs` is the video
callback's `metadata.mediaTime * 1000`, retaining fractional precision. A manual
camera snapshot may use its preview capture timestamp when video-frame
callbacks are unavailable; the copied pixels and identity stay together.
Return one canonical [FrameResult](../contracts/schemas/frame-result.schema.json)
with all five identity fields unchanged and `coordinateSpace: "original_pixels"`.
The Python runner produces direct `ml_prediction` results; the frontend contract
also accepts correctly identified `propagated_prediction` results with provenance.
The model identity must match the readiness record. Reviewed annotations and
synthetic fixtures cannot masquerade as live model output.

The camera picker requests 1280 x 720 with each dimension capped at 1920,
so high-resolution capture devices negotiate a supported stream size. Camera,
uploaded-image, uploaded-video and stream captures fit within 1280 x 720, preserving aspect
ratio without upscaling. Frame identity and overlay geometry use these actual
captured dimensions; the source video may have a higher resolution. The protocol
limits a capture to 4096 pixels per side, 4,194,304 total pixels and a
3 MiB JPEG in the browser, keeping base64 JSON below Vercel's 4.5 MB request
limit. The internal bridge and runner cap JSON at 6 MiB and results at 2 MiB. The runner
validates actual decoded dimensions, refuses rotated EXIF captures, and rejects
concurrent inference with HTTP 429. Invalid or unavailable results clear anatomy.
The bridge preserves a sanitized 429 response. Continuous video identification
displays **Identification is busy** and waits two seconds before accepting another
capture, avoiding video-rate retries against an occupied model. Paused video
frames retain their bounded retry limit; if retries are exhausted, use Play or
Refresh model. Manual camera identification does not retry automatically;
use **Retake**, then **Capture & identify** to try again. Both paths wait for an
outstanding request to settle before submitting another, even after cancellation.
An `ok` result with no structures is distinct from failed identification.

Camera identification submits one request per **Capture & identify** action;
uploaded images require **Identify image**. Neither samples repeatedly while
previewing, waiting for a response or showing a result. Uploaded videos and
stream links attempt at most five captures per second with one request in flight.
All input modes allow **6,000 ms
from capture** for JPEG encoding, network transit and inference. Every accepted
result appears with the exact captured pixels that produced it.

The completed camera snapshot remains available until retaken or its session
is reset. It is visibly a frozen capture, not a current camera overlay. The
continuous-video path expires old results and returns to preview with anatomy
cleared, except during deliberate paused-frame inspection. Continuous video
interruptions, dimension changes and backward media timestamps invalidate old
work. A frozen camera snapshot keeps its copied pixels through native camera
dimension or timestamp changes; camera interruption or a page visibility change
still clears it. Browsers
without video-frame callbacks support manual camera snapshots and paused-video
identification; playing-video identification requires those callbacks. Actual
throughput and alignment need device validation.

The promoted checkpoint was installed and strictly loaded on the Windows CPU
host on September 14. Nine HTTP predictions using the existing training
reference image at 854 x 480, 1280 x 720 and 1920 x 1080 took 1,540–1,937 ms.
Three further 1280 x 720 requests through Vite's `/api/identify` took
1,618–1,658 ms and passed the shared result schema and exact frame identity
checks. These are connection and speed checks, not accuracy measurements or
phone camera tests. The former 750 ms deadline rejected every measured result;
the six-second bound leaves room for CPU inference, network transit and display
of the matching captured image. Continuous uploaded-video and stream updates
on this host will be much slower than the five-captures-per-second ceiling;
manual camera captures wait for one result instead of scheduling the next frame.

The September 14 production deployment at `https://holospex.vercel.app/mannequin`
was also checked through Vercel and the authenticated Cloudflare tunnel. Three
generated 1280 x 720 gray-grid images returned canonical model results with
matching frame identities in 1,620–1,814 ms. All returned zero structures.
The deployed JavaScript and CSS matched the tested local build byte for byte.
This verifies the public connection and timing; actual phone-camera operation
and anatomical accuracy still require device testing.

Low-confidence or unsupported anatomy is hidden. Partial visibility has dashed
outlines; warnings say **Unable to identify** and remain distinct from labels.
Backup recordings capture the composed canvas and stay in the browser. Stop
and save the recording before switching modes.

## Pointer clarity and model limits

Each anatomical class now has one pointer inside its largest predicted
polygon. Smaller components retain their supplied fills and outlines without
duplicating the class name in the label rail. The old average
of polygon vertices could fall outside a concave region, making a correctly
drawn mask point at neighboring tissue. This fixes label placement without
changing the model's predicted boundary; the pointer is a geometric label
position, not a separately identified anatomical landmark. **Overlay appearance** offers
**Masks + outlines** and **Outlines only**; the latter leaves tissue texture more
visible. **Show model scores** reveals component scores when useful for
inspection. These display controls do not change identification or certainty.

The model's **default per-pixel cutoff is 0.5**. The applied percentage controls
both per-pixel filtering in the runner and confidence filtering in the overlay.
Each request carries its own cutoff without changing another user's settings
or the model's export defaults. Exported component
scores are the mean softmax of pixels that survive the cutoff, and are not
calibrated probabilities of anatomical correctness. Increasing the cutoff
can hide more anatomy; reducing it can admit additional false positives.
The polygon exporter also withholds components with holes, non-simple
boundaries, or contour area below 64 captured pixels. Missing labels do not
necessarily mean the raw model predicted no anatomy.

The [current model record](../ml/CURRENT_MODEL.md) reports 52.8251% six-class
foreground IoU and 35.7053% small-anatomy IoU on its fixed 75-frame native
validation set. Those are recorded validation metrics, not measured accuracy
on this camera or uploaded clip. No held-out image/mask set is installed in
this checkout for a fresh comparison. The pointer and inspection changes have
no demonstrated model-accuracy gain; actual phone interaction, stream
playback and alignment still require device testing.

Use the original laparoscopic surgical video or direct surgical feed when
available, keeping the operative field in view. Filming a screen adds glare,
perspective and another resampling step. The model was trained on surgical
imagery; an external mannequin or room view does not provide the internal
anatomy it was trained to segment. A faster inference host can reduce visible
delay, but it does not establish better segmentation accuracy.

## Frontend integration

`LiveFeedDemo` owns camera selection, uploaded image/video files, stream links, readiness
and recording controls. `LiveFeedHud` owns the manual **Capture & identify** /
**Retake** controls and the frozen camera result. `UploadedImageHud` owns local
image decoding, explicit **Identify image** and persistent image results.
URL playback uses browser media loading and the
optional HLS adapter; the model bridge and frame contract stay unchanged.
A custom frontend can use the same renderer:

```tsx
<LiveFeedHud
  stream={stream}
  interrupted={cameraInterrupted}
  visible={overlaysRequested}
  identify={ready ? identifyLiveFrame : undefined}
  minimumConfidence={modelThreshold}
  onDisplayedFrame={setDisplayedFrame}
  onCanvas={setRecordingCanvas}
/>
```

`input/liveIdentification.ts` exports `loadIdentificationModel(signal)`,
`identifyLiveFrame(frame, jpegBlob, signal)`, `validateLiveResult(value, frame)`
and the `LiveFrameIdentifier` type. Injected identifiers use the same validated
result contract. `onDisplayedFrame` reports the displayed frame or `null` when
cleared. Live identification does not own lesson scoring or response recording;
those controls remain in the recorded lesson/reference workflows.

Automated tests exercise the client, private bridge and Python runner with
explicit test predictors. Those tests verify the connection and failure
behavior, not trained-model accuracy, camera compatibility or real-device latency.

## Training image references

Select **Training image AR** for labeled stills on the camera/mannequin, or
open `/samples` to inspect their exact annotation masks. These views accept
training-split references only, including manual and portable imports;
validation and test data are excluded without relabeling them.

Prepare a small batch from the existing ML manifest and original image/mask
files. The exporter requires Pillow and performs no model inference:

```sh
python scripts/prepare-training-reference.py --manifest ml/outputs/endoscapes-manifest.json --limit 12
npm run dev:samples
```

The default output is ignored `runs/training-reference/`. Use `--offset 12`
and a fresh `--output` directory for another batch; use `--data-root` if the
manifest's dataset has been relocated while retaining its relative layout.
Missing training data produces an unavailable state. The browser never
falls back to `apps/web/tests/samples_tst`.

In this opt-in development mode, `/__local-training/` serves only validated
training references from the prepared folder. `HOLOSPEX_TRAINING_REFERENCE_DIR`
can select another prepared folder on the server; it is not a public Vite
environment variable. The endpoint is absent from production builds. Use
**Reload training images** after preparing a batch.

For a phone, run `npm run prepare:camera-samples` against the prepared folder,
transfer a generated `.holospex.json` from `runs/camera-samples/` to Files,
then open it in Training image AR. An alternate folder can be passed as
`npm run prepare:camera-samples -- runs/another-training-batch`. Training
images, masks and portable files stay out of Git and public deployment.
See [the image overlay guide](camera-image-overlay.md) for placement controls.
