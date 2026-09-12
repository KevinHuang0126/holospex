# Integration assets

These JSON records and the SVG were authored for this scaffold. The illustration
is abstract geometry, not a patient image or anatomical model. All sample
provenance and answers are `synthetic_mock`; no clinician has reviewed them.
They inherit this repository's license.

`lesson.json` links two checkpoints: a visible mocked polygon and an unsupported
result. They exercise rendering, absence of overlays, feedback and attempts.
The example CVS question checks that synthetic shapes cannot establish a real
surgical assessment; it is not an expert-authored clinical lesson.

The root preparation script copies only named fixture files. To add a real
lesson, record its origin/license, permission for the demo, media timebase and
annotation cadence, plus the actual geometry/answer review provenance here.
Update the explicit publication list deliberately; do not recursively copy a
dataset into public assets. Store datasets under ignored `data/`, weights under
`models/`, and pipeline outputs under `runs/` until selected for integration.
