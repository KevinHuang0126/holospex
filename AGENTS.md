# Working on Holospex

Read `README.md` and `docs/architecture.md` before changing module boundaries.
This is a two-day educational hackathon prototype with a future AR-glasses
direction. The web scaffold uses synthetic geometry. The ML package now has a
real Endoscapes training and export path; read `ml/TRAINING.md` and `ml/STATUS.md`
before describing its capabilities. The user leads ML and general
architecture. Dataset labels and predictions are not reviewed lesson answers.

## Ownership and parallel work

- Person 1 / ML lead: `ml/`, `contracts/`, root tooling, integration assets.
- Person 2 / AR: `apps/web/src/overlays/`, `apps/web/src/camera/`, camera input.
- Person 3 / frontend: lesson flow, analytics, app shell, styles in `apps/web/`.
- Inspect current files and work in progress before editing. Keep another
  contributor's changes. Coordinate shared contracts; do not independently
  rename fields, anatomy IDs, or exported controls.
- Prefer the smallest complete handoff. Do not introduce an API server,
  database, authentication, a glasses SDK, or heavy model dependencies unless
  the requested feature needs them. Routine fixes do not need extra approval.

## Boundaries to preserve

1. JSON Schemas in `contracts/schemas/` define the wire format. Generated
   TypeScript is derived, never hand-edited. Python validates the same schemas.
2. Keep reviewed lesson answers separate from frame predictions. An ML score,
   missing prediction, or tracking failure cannot determine a CVS answer.
3. Keep surgical-video segmentation separate from physical-model marker
   registration. Original image pixels and world/model coordinates are not
   interchangeable. Glasses need the laparoscope feed to show internal imagery.
4. Keep source labels explicit: synthetic, reviewed, predicted, propagated.
   Never relabel synthetic/model output as reviewed to complete a demo.
5. Clear overlays when media/frame identity does not match or tracking is lost.
   Do not hold a sparse annotation over unrelated video frames.
6. Store elapsed time and cumulative hint exposure with a submitted answer.
   Feedback follows commitment; renderers do not own correctness or scoring.
7. Keep datasets, weights, recordings, credentials, and learner exports out of
   Git. Only add media to the explicit demo asset list after checking its use
   and recording provenance. No automatic dataset downloads or clinical-use claims.

## Checks and handoff

- Contract change: regenerate with `npm run contracts:generate`, update the
  producer/consumer and fixtures, then run `npm run check` and Python tests.
- Web change: `npm run check` and `npm run build`. Exercise the affected flow
  in a browser when interaction or rendering changes.
- ML change: `.venv/bin/python -m unittest discover -s ml/tests -v` after
  installing the optional dependencies needed for the changed modules.
- Run only checks relevant to the change; avoid tests that merely restate code.
- Report what works, what was tested, and which adapters remain placeholders.
  Do not describe a directory, interface, or mock as a completed capability.

Unresolved details are tracked in `docs/team-plan.md`; continue independent
scaffold work while awaiting compute, real data, hardware, and reviewer choices.
