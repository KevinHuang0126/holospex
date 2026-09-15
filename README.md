# Holospex

AR/HUD infrastructure is now available for teammate integration. See the
[renderer and camera handoff](docs/ar-hud-infrastructure.md) for public components,
result inputs, model configuration, controls, tests, and remaining device checks.

A shared starting point for a two-day surgical education hackathon: a browser
lesson and a phone/desktop camera path, designed with future AR glasses in mind.

**Web scaffold:** a synthetic image-checkpoint lesson, validated data handoffs,
canvas video HUD, frontend controls, calibrated marker-registration modules,
camera input, recording support, and answer export. The **ML package** provides
real Endoscapes data acquisition,
preparation, model training, evaluation, and inference exports. See the
[ML runbook](ml/TRAINING.md) and [measured run results](ml/STATUS.md).
Trained weights stay outside Git and the public web bundle; surgical data remains
local and the web lesson still uses
synthetic fixtures. **Camera identification**, **Upload image**, **Upload video** and **Stream link**
identify anatomy through the built-in connection to Person 1's trained model.
**Training image AR** and `/samples`
load prepared train-split stills and exact masks with dataset provenance.
See the [live-feed and training-reference guide](docs/live-feed.md).
The cloud demo hosts the pinned model on Cloud Run; local training assets and
device validation still need their actual inputs.
Reviewed lesson content and glasses integration remain future work.

**Current ML model:** the September 14 weight-decay-0.05 DeepLabV3–ResNet50 checkpoint
at `ml/weights/current/best.pt` leads the saved native-validation comparisons
with **53.3060% foreground IoU**. See the [current model handoff](ml/CURRENT_MODEL.md)
for its exact identity, verified artifact retrieval, inference command and
video-export compatibility. A Git checkout does not include the weights or
prediction exports; the browser's bundled lesson remains synthetic.

For live identification, install the ML virtual environment, then run
`npm run model:prepare -- --checkpoint PATH_TO_DOWNLOADED_BEST_PT` and
`npm run model:serve` in one terminal, with `npm run dev` in another.
The installer and default runner verify Person 1's promoted checkpoint checksum;
the live runner uses its recorded ResNet50 preprocessing and a default 0.5 cutoff.
Enter a **Confidence cutoff (%)** and select **Apply cutoff** to use your own
value for camera captures, uploaded images or videos, or stream links.
See [checkpoint setup and phone hosting](docs/live-feed.md).

## Start the browser app

For phone camera testing, open the [deployed identification demo](https://holospex-mu.vercel.app/mannequin)
in your phone browser and allow camera access. This deployment uses the selected
model on Cloud Run and needs no laptop. After inactivity, allow time to start and
press **Refresh model** if needed. See the [cloud runbook](ml/CLOUD_INFERENCE.md).
See the [phone setup and redeployment instructions](docs/mannequin-overlay.md#open-it-on-a-phone).

The live camera/mannequin overlay is at **http://127.0.0.1:5174/mannequin**
when running `npm run dev:samples` (or `/mannequin` on the URL from `npm run dev`).
The default **Camera identification** previews a camera or USB capture device
and checks model readiness automatically. **Start camera** opens the live preview.
Once Person 1's checkpoint runner is available, **Capture & identify** sends one
displayed frame and freezes it with its matching anatomy result. **Retake**
returns to preview; another explicit capture is needed to send the next frame.
While the model is unavailable, camera preview stays usable.
The Python runner and private `/api/identify` bridge are implemented; weights
are not bundled. Identification allows one request in flight and 6,000 ms from
capture for encoding, transit and inference. A completed camera snapshot stays
visible until retaken or its session resets. **Upload image** previews a local
surgical image; press **Identify image** to send it to the model and retain its
matching result. Selecting a file does not send it automatically. Applying a new
cutoff clears the result while keeping the image for another explicit identification.
**Upload video** uses the same model
to identify sampled frames continuously from a local clip,
with play, pause, seek and replay controls; it needs no result JSON or camera
permission. **Stream link** accepts a direct HTTPS video or HLS URL with
cross-origin access enabled by its host. Choose the link format and press
**Connect stream**; use **Reconnect stream** after changing the link or format,
and **Disconnect stream** to release it. Only sampled frames go to the model.
See [stream requirements](docs/live-feed.md#identify-a-stream-link) and
[model startup and hosting](docs/live-feed.md).

Select **Training image AR** to place a labeled training still on the camera
or supplied mannequin cutout. **Camera screen** needs no marker; **Table marker**
anchors the image and labels together. **Scene → Mannequin + sample** supports
manual placement and the separate [placement API](docs/anatomy-placement-api.md).
These annotations remain training references, not live inference or reviewed
answers. Marker tests and measured-model configuration remain available.

To prepare and inspect training references, first supply the existing ML
manifest and its original JPEG/mask files, then run:

```sh
python scripts/prepare-training-reference.py --manifest ml/outputs/endoscapes-manifest.json --limit 12
npm run dev:samples
```

Open **http://127.0.0.1:5174/samples**, or choose **Training image AR** on
`/mannequin`. **Reload training images** reads ignored `runs/training-reference/`;
the server-only `HOLOSPEX_TRAINING_REFERENCE_DIR` can select another prepared
folder. Automatic and manual imports exclude validation/test splits, with no
fallback to `samples_tst`. The private `/__local-training/` endpoint exists only
in this opt-in development mode. Missing training assets remain unavailable.

To transfer a prepared training reference to a phone, run
`npm run prepare:camera-samples` and open a generated `.holospex.json` from
`runs/camera-samples/` in Training image AR. Images and masks are excluded from
public deployment. See [preparation and batch options](docs/live-feed.md#training-image-references)
and the [image controls](docs/camera-image-overlay.md).

Use Node 22.12+ (Node 24 LTS is a suitable team baseline) and npm.

```sh
npm ci
npm run dev
```

Open the localhost URL printed by Vite. The development server listens on the
LAN for device testing. Camera access requires a secure browser context:
desktop localhost works, but a phone visiting your computer's plain HTTP LAN
address generally cannot use the camera. Use an agreed HTTPS development host
for the phone demo. Camera preview starts only after an explicit button press.

```sh
npm run check
npm run build
```

The build is emitted to `apps/web/dist/`. Preview and local image rendering need
no model runtime; live identification requires the configured checkpoint runner.
`npm run prepare:demo` validates and copies a fixed allowlist of
synthetic source assets and the selected mannequin reference from
`assets/demo/` into `apps/web/public/demo/`.
The copy is generated: edit `assets/demo/`, not `public/demo/`.

## Start the Python pipeline

Use Python 3.11+ in a virtual environment; heavy ML libraries are deliberately
left to the selected model adapter.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./ml
python -m holospex_ml validate assets/demo/frame-000.json
PYTHONPATH=ml/src python -m unittest discover -s ml/tests -v
```

For the actual dataset and training workflow, install `./ml[train]` instead and
follow [`ml/TRAINING.md`](ml/TRAINING.md). The lightweight unconfigured adapter
is retained to test explicit unavailable-result handling.

## Where to work

| Area | Entry point | Owner |
| --- | --- | --- |
| ML and preprocessing | [`ml/`](ml/) | ML/architecture lead |
| Canonical wire contracts | [`contracts/README.md`](contracts/README.md) | Lead, with consumers |
| Video overlays and camera AR | [`apps/web/`](apps/web/) | AR teammate |
| Lesson, feedback, attempts | [`apps/web/`](apps/web/) | Frontend teammate |
| Integration fixtures | [`assets/demo/README.md`](assets/demo/README.md) | Shared |
| Boundaries and route to glasses | [`docs/architecture.md`](docs/architecture.md) | Everyone |
| Handoffs and two-day gates | [`docs/team-plan.md`](docs/team-plan.md) | Everyone |
| Agent working guidance | [`AGENTS.md`](AGENTS.md) | Agents and contributors |

The JSON Schemas are authoritative. Run `npm run contracts:generate` after a
schema edit; `npm run contracts:check` detects stale generated types. Browser
parsers and Python validation reject invalid geometry/provenance at boundaries.
Passing validation checks structure and consistency, not anatomical accuracy.

Browser attempts stay on that device and can be exported. They are demo data,
not a durable research database. Never commit learner exports or patient media.

Framework references: [Vite setup](https://vite.dev/guide/),
[Ajv TypeScript validation](https://ajv.js.org/guide/typescript.html).
