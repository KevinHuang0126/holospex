# Integration assets

These JSON records and the SVG were authored for this scaffold. The illustration
is abstract geometry, not a patient image or anatomical model. All sample
provenance and answers are `synthetic_mock`; no clinician has reviewed them.
They inherit this repository's license.

`lesson.json` links two checkpoints: a visible mocked polygon and an unsupported
result. They exercise rendering, absence of overlays, feedback and attempts.
The example CVS question checks that synthetic shapes cannot establish a real
surgical assessment; it is not an expert-authored clinical lesson.

`mannequin.png` is a user-provided reference image, copied byte-for-byte from
`apps/web/tests/helpers/mannequin.png` after the user requested its inclusion
as the background of the camera composite. It shows an anatomical mannequin
against a white background. Its original creator, external source and license
have not been verified; it is not covered by the scaffold-authored provenance
or license statement above. No anatomical or clinical review is claimed.
The publication allowlist includes its transparent derivative below, not the
helpers directory or surgical dataset images.

- Format and size: PNG, 894 × 569 pixels, 410,943 bytes.
- SHA-256: `cbeb22f2c521d8ecad5be3ed9061bd3d707c065c01b02dbf5eb426f38dbf4c51`.
- Original reference is retained locally in this directory.

`mannequin-cutout.png` is derived from that reference with code, as requested by
the user. `scripts/prepare-mannequin-cutout.py` removes the white backdrop and
softens the cutout edge. It keeps the original 894 by 569 pixel canvas and body
positions, so sample and label coordinates remain aligned. This changes only
the display reference; surgical images and categorical masks are unaffected.
The derivative has the same provenance and unverified external license as the
original; no new anatomical review is implied. To regenerate it locally, run
`python scripts/prepare-mannequin-cutout.py` with Pillow installed. Normal web
builds copy the prepared PNG and require no Python or background-removal API.

- Public path: `/demo/mannequin-cutout.png`.
- Format and size: RGBA PNG, 894 by 569 pixels, 380,242 bytes.
- SHA-256: `c25ddd6715d6402786780008588763b56d9dc970dc438bf3256447d7347f12c1`.
- Every RGB byte is preserved; 369,231 backdrop pixels become fully transparent
  and 1,968 edge pixels have partial alpha. Minor thin edge fringes remain.

The root preparation script copies only named fixture files. To add a real
lesson, record its origin/license, permission for the demo, media timebase and
annotation cadence, plus the actual geometry/answer review provenance here.
Update the explicit publication list deliberately; do not recursively copy a
dataset into public assets. Store datasets under ignored `data/`, weights under
`models/`, and pipeline outputs under `runs/` until selected for integration.
