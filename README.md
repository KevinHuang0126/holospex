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
Trained weights and surgical data remain local; the web lesson still uses
synthetic fixtures. The **Dataset samples** view imports Person 1's local
Endoscapes stills and exact masks with explicit dataset provenance. See the
[sample connection handoff](docs/dataset-sample-connection.md) for loading
`samples_tst` and connecting frontend callbacks. Video prediction integration,
model calibration, and device validation still need their actual inputs.
Reviewed lesson content and glasses integration remain future work.

**Current ML model:** the September 14 batch-002 DeepLabV3–ResNet50 checkpoint
at `ml/weights/current/best.pt` leads the saved native-validation comparisons
with **52.8251% foreground IoU**. See the [current model handoff](ml/CURRENT_MODEL.md)
for its exact identity, verified artifact retrieval, inference command and
matching video predictions. A Git checkout does not include the weights or
prediction exports; the browser's bundled lesson remains synthetic.

## Start the browser app

For phone camera testing, open the [deployed mannequin demo](https://holospex-mannequin-phone.vercel.app/mannequin)
in your phone browser and allow camera access. The laptop can be turned off.
See the [phone setup and redeployment instructions](docs/mannequin-overlay.md#open-it-on-a-phone).

The live camera/mannequin overlay is at **http://127.0.0.1:5174/mannequin**
when running `npm run dev:samples` (or `/mannequin` on the URL from `npm run dev`).
The default **Labeled image** overlays the original surgical JPEG and its exact
anatomy mask on the camera. **Camera screen** keeps the image fixed on screen;
**Table marker** places it as a flat image beyond a printed marker. Choose
**Full surgical image** or **Anatomy cutout**, then adjust size and opacity.
**Scene → Mannequin + sample** places that image on the supplied mannequin
photo. **Fit anatomy with AI** requests a position and scale from the
[placement API](docs/anatomy-placement-api.md); sliders allow fine adjustment.
Use **Table marker** to anchor the mannequin, image and labels together.
The composite stays flat, and its illustrative placement needs visual checking.
This preserves supplied dataset annotation provenance; it does not reconstruct
3D anatomy or run live inference. Learn reveals the image and labels;
Identify and Assess hide them. Feedback is withheld because reviewed answers
have not been supplied.
Local sample mode loads the usable `samples_tst` cases automatically. To use a
sample on your phone, run `npm run prepare:camera-samples`, transfer one generated
`.holospex.json` file from `runs/camera-samples/` to the phone's Files app, and
select it under **Open camera sample or sample files**. Dataset files are excluded
from the public deployment. **Marker test** and **Mannequin configuration** remain
available for tracking checks and measured model locations. See the
[image overlay guide](docs/camera-image-overlay.md) and
[camera setup guide](docs/mannequin-overlay.md).

To test the current samples independently of the learning frontend:

```sh
npm run dev:samples
```

Open **http://127.0.0.1:5174/samples**. The standalone React page automatically
loads the usable cases from `apps/web/tests/samples_tst`, with case navigation,
overlay controls, learning modes and a click-selection readout. **Reload local
samples** re-reads that folder. Its sample endpoint is local to this opt-in
development command; production builds retain the manual folder picker.

The tester now focuses on the supplied image overlays. Adjust fill opacity,
switch boundaries and labels independently, or use **Boundaries only** to
inspect alignment. **Show overlays** instantly returns to the original image.
Identify and Assess override all layer controls and keep anatomical answers hidden.

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

The build is emitted to `apps/web/dist/`. No inference server, account, or API
key is needed. `npm run prepare:demo` validates and copies a fixed allowlist of
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
