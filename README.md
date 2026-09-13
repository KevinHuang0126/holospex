# Holospex

AR/HUD infrastructure is now available for teammate integration. See the
[renderer and camera handoff](docs/ar-hud-infrastructure.md) for public components,
result inputs, model configuration, controls, tests, and remaining device checks.

A shared starting point for a two-day surgical education hackathon: a browser
lesson and a phone/desktop camera path, designed with future AR glasses in mind.

**Current scaffold:** a synthetic image-checkpoint lesson, validated data handoffs,
canvas video HUD, frontend controls, calibrated marker-registration modules,
camera input, recording support, answer export, and a Python inference boundary.
Real surgical assets, model calibration and device validation are still needed.
There is no trained surgical model, real reviewed lesson, or glasses integration.

## Start the browser app

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
synthetic source assets from `assets/demo/` into `apps/web/public/demo/`.
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

See [`ml/README.md`](ml/README.md) for the unconfigured-adapter export command,
geometry mapping, and where to add the real model. The unconfigured adapter
returns an explicit unsupported result; it does not invent detections.

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
