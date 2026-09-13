# Anatomy placement API

In `/mannequin`, load a camera sample, choose **Mannequin + sample**, and select
**Fit anatomy with AI** in Learn mode. The app requests a body region, position
and size, applies them to the sample, and keeps the existing position/size
sliders available for adjustment. Switching samples discards the previous
suggestion; switching sources, manual edits and unmounting cancel pending work.
The button makes one request; tracking frames and slider movement make none.

## Setup

Add `OPENAI_API_KEY` to the Vercel project's Production environment, then deploy
again. The existing `OPEN_AI_KEY` name is also accepted; `OPENAI_API_KEY` takes
precedence if both are present. The optional `OPENAI_PLACEMENT_MODEL` overrides the default
`gpt-4.1-mini-2025-04-14`. These variables are server-only; never prefix the key
with `VITE_`, put it in a sample file, or commit it. The endpoint returns a clear
503 error when its key is missing; it does not claim a local preset is AI output.

For local development, use a root `.env.local` file (Git-ignored) or set the
environment variables in the terminal that starts `npm run dev:samples`.
Restart that server after changing the environment. The Vite middleware handles
`/api/placement` during development. Static `vite preview` does not run the API.

Phone deployment still uses `npm run build`, `npm run prepare:phone`, then
`vercel deploy --cwd runs/holospex-mannequin-phone --yes --prod`.
The packaging script places the bundled Node function outside `public/`.
Surgical samples and secrets stay outside the published asset list.

## Geometry and data flow

1. The importer validates the sample's original image and exact index mask.
2. `imageOverlayAsset.ts` measures each present anatomy class directly from the
   raster, including components withheld from polygon conversion. Bounds use
   normalized pixel edges; centroids average pixel centers. Background, tools
   and ignored pixels do not become anatomy measurements.
3. `POST /api/placement` accepts only sample/template IDs, dimensions and these
   class measurements. No image bytes, camera frames or recordings are sent.
4. The server calls the OpenAI Responses API with `store: false` and a strict
   structured output schema. It supplies authored landmarks for this specific
   mannequin. Returned locations and sizes must be within the selected region;
   unknown, contradictory and ambiguous labels fail explicitly.
5. `samplePlacement.ts` scales the labeled focus's bounding box and moves its
   centroid to the suggested target. It applies that single transformation to
   the original image, mask and all label anchors. If the full frame would leave
   the reference image, it reduces scale while retaining the target centroid.
6. The existing renderer projects the composite onto the marker plane. Marker
   loss and camera interruption retain the existing overlay-clearing behavior.

The mannequin's head points left and its feet right; its anatomical right is
toward the bottom of the image. Gallbladder-class imagery targets the upper
right abdomen. This anatomical association is consistent with the
[NIDDK description](https://www.niddk.nih.gov/health-information/digestive-diseases/gallstones/definition-facts).
The exact template coordinates and scale ranges are authored illustrative
values, not measured anatomical registration or clinical validation.

The metadata contract also accepts brain-related classes and targets the head.
That route is tested using synthetic metadata; the existing `.holospex.json`
file importer still accepts Endoscapes samples. A brain MRI model needs its own
validated image/mask adapter before its images can be loaded in this UI. No
existing ML class IDs or generated contracts were changed for this feature.

## Limits and verification

The endpoint limits body size, checks same-origin browser requests, bounds its
provider timeout and output, and returns sanitized failures. Its concurrency
and hourly request guard are per server instance. They are demo safeguards,
not a global spending cap across serverless instances; configure project-level
provider budgets for a broader deployment.

The production endpoint was deployed and checked on 2026-09-13 with the prepared
`116_35325` sample's metadata. It found the production key, but OpenAI returned
HTTP 429. The UI reports `placement_quota`; this can mean a temporary rate limit
or unavailable API quota. Live AI placement remains unverified until that limit
is resolved. See [OpenAI's 429 troubleshooting guide](https://help.openai.com/en/articles/5955604).
Local API, geometry and cancellation tests pass. Published application assets
and the exclusion of private sample routes were checked; camera alignment still
needs a test on the actual phone.

AI suggestions do not change the sample's provenance or make it a reviewed
lesson answer. The composite remains a flat image with approximate camera
calibration. Fine adjustment on the actual device is still required.

API implementation references: [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
and [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
