# Cloud identification deployment

Verified September 14, 2026 (September 15 UTC). The frontend is in Kevin's
Vercel account and the selected model runs on Google Cloud Run. No laptop or
Cloudflare tunnel is needed for this deployment.

```text
Browser: holospex-mu.vercel.app
  -> same-origin /api/identify (Vercel Node function)
  -> HTTPS /identify + server-only bearer token (Cloud Run)
  -> pinned PyTorch checkpoint -> canonical FrameResult -> browser
```

## Deployed resources

| Item | Value |
| --- | --- |
| App | https://holospex-mu.vercel.app/mannequin |
| Vercel project | `kevhuang-3216s-projects/holospex` |
| Model endpoint | `https://holospex-identification-576811516435.us-central1.run.app/identify` |
| Google project / region | `eastwest72hack26bos-501` / `us-central1` |
| Cloud Run service / initial revision | `holospex-identification` / `holospex-identification-00001-l5q` |
| Resources | 4 vCPU, 8 GiB RAM, concurrency 1, minimum 0, maximum 1 |
| Runtime service account | `holospex-identification@eastwest72hack26bos-501.iam.gserviceaccount.com` |
| Secret | `holospex-identification-token`, pinned version `1` |
| Cloud Build | `aa9f59c5-1f9d-49e7-8e95-09932ca09c36` |
| Model version | `2026-09-14T16:39:12.476979Z-epoch-43` |
| Checkpoint SHA-256 | `9dc50d58fb2f605f5fc2a00652d7dae662f7584ab08e37502fb179b152873c1b` |

The initial image is
`us-central1-docker.pkg.dev/eastwest72hack26bos-501/holospex-inference/model@sha256:157fc11232feab7016d07a85ecf1a3d13531d411b599621f43ff57da41f0704c`.
It contains ML source from commit `64bb6888d5ed679a0ba57cfbc6925a2bd5a7e5db`
and the explicit deployment files recorded in the local context manifest.
The older `holospex.vercel.app` belongs to another account and was not changed.

The runtime account can read only the named secret. The token is absent from
the browser bundle. The public Vercel bridge remains an unauthenticated demo;
same-origin checks are not user authentication or a complete abuse defense.
Maximum-instance settings limit intended scaling, not total spend. Stop the
feed or close the tab when finished: frame requests can keep an instance active.
Image storage, builds and request processing can incur charges. Do not expose
real patient data through this prototype.

## Verified behavior and limitations

- Browser displays **Identification model ready** on the new production URL.
- Direct requests without the token return HTTP 401; cross-origin bridge
  requests return HTTP 403.
- Readiness returns the pinned model version and threshold 0.5.
- Two generated 672 x 384 gray-grid frames through Cloud Run returned HTTP 200
  in 4.03 and 2.34 seconds; one through Vercel returned HTTP 200 in 2.86 seconds.
  All passed the shared schema and exact frame identity checks, with no
  structures. These are connection checks, not accuracy or FPS benchmarks.
- All 358 ML tests, `npm run check`, `npm run build` and `npm run prepare:phone`
  passed. Actual camera/video performance and anatomical accuracy on the target
  device still require testing.

Minimum zero allows the model to stop after inactivity. The first rollout took
about 19 seconds from instance startup to listening, longer than the bridge's
10-second timeout. Cold-start latency varies. If unavailable after an idle
period, wait about 20–30 seconds and press **Refresh model**. Once ready, use
**Upload video** or **Start camera**. The six-second frame expiry still applies;
stale predictions must not remain on unrelated frames. Several users can
contend for this single CPU instance.

The separate placement API needs its own OpenAI configuration. Training-image
references, reviewed lesson answers and clinical validation are not supplied by
this deployment. Model selection and data/license limitations remain documented
in [CURRENT_MODEL.md](CURRENT_MODEL.md) and [TRAINING.md](TRAINING.md).

## Rebuild the model container

Run from the repository root with selected weights installed and the `holospex`
Google Cloud configuration authenticated. Use a fresh run name; the packager
refuses to overwrite a context. It copies only tracked Python source, schemas,
explicit container files and verified weights. It excludes datasets, recordings,
`.env` files and credentials. `context-manifest.json` records file hashes.

```sh
RUN_ID="inference-$(date -u +%Y%m%dT%H%M%SZ)"
IMAGE="us-central1-docker.pkg.dev/eastwest72hack26bos-501/holospex-inference/model:$RUN_ID"
.venv/bin/python ml/cloud/prepare_inference.py --output "runs/$RUN_ID/context"
gcloud --configuration=holospex builds submit "runs/$RUN_ID/context" \
  --project=eastwest72hack26bos-501 --region=us-central1 \
  --tag="$IMAGE" --timeout=1200
```

Wait for `SUCCESS` and record the immutable image digest from the build result.
Set `IMAGE_DIGEST` to that full registry `image@sha256:...` reference:

```sh
gcloud --configuration=holospex run deploy holospex-identification \
  --project=eastwest72hack26bos-501 --region=us-central1 \
  --image="$IMAGE_DIGEST" \
  --service-account=holospex-identification@eastwest72hack26bos-501.iam.gserviceaccount.com \
  --set-secrets=HOLOSPEX_IDENTIFICATION_TOKEN=holospex-identification-token:1 \
  --cpu=4 --memory=8Gi --concurrency=1 --min=0 --max=1 --max-instances=1 \
  --port=8080 --timeout=30 --execution-environment=gen2 \
  --cpu-boost --allow-unauthenticated
```

The registry, service account, secret and scoped grant already exist. Do not
regenerate the token during routine deployment. Rotation requires coordinating
a new Secret Manager version and Vercel production variable, then redeploying
both. Never paste the token into source, screenshots, URLs or CLI arguments.

## Redeploy the frontend

The GitHub connection attempt failed, so pushing `main` does **not** automatically
deploy this Vercel project. Use the prepared-directory workflow. Production
variables `HOLOSPEX_IDENTIFICATION_URL` and `HOLOSPEX_IDENTIFICATION_TOKEN`
are already set server-side.

```sh
npm run check
npm run build
npm run prepare:phone
vercel login
# On a new checkout only; verify this links to your account's project:
vercel link --yes --scope kevhuang-3216s-projects --project holospex \
  --cwd runs/holospex-mannequin-phone
vercel deploy --prod --yes --cwd runs/holospex-mannequin-phone
```

`prepare:phone` preserves the local `.vercel` project link. Its upload allowlist
contains generated public assets, bundled APIs and package/config files. Python
and the checkpoint are deployed separately to Cloud Run and remain outside Git.

## Check status and troubleshoot

```sh
gcloud --configuration=holospex run services describe holospex-identification \
  --project=eastwest72hack26bos-501 --region=us-central1 \
  --format='yaml(status.conditions,status.latestReadyRevisionName,status.url)'
gcloud --configuration=holospex run services logs read holospex-identification \
  --project=eastwest72hack26bos-501 --region=us-central1 --limit=30
curl --fail-with-body https://holospex-mu.vercel.app/api/identify \
  -H 'Origin: https://holospex-mu.vercel.app' -H 'Sec-Fetch-Site: same-origin'
```

Opening `/api/identify` in the address bar lacks same-origin fetch metadata and
may return 403. Use the command above or **Refresh model** in the app. Expected
readiness is `status: ready` with the epoch-43 version above. A 503/504 may mean
startup, a mismatched token or unavailable revision; 429 indicates contention.
Check Cloud Run logs and Vercel variables before rebuilding the model.

Local receipts and synthetic smoke results are under ignored
`runs/cloud-deployment-20260915/`. Its private token file is mode 0600; do not
share that directory wholesale. Secret Manager is the durable credential source.
Keep the service enabled for the demo; deleting it stops inference.

Provider references: [Cloud Run container contract](https://docs.cloud.google.com/run/docs/container-contract),
[secret integration](https://docs.cloud.google.com/run/docs/configuring/services/secrets),
[Cloud Run pricing](https://cloud.google.com/run/pricing),
[Vercel deployment](https://vercel.com/docs/cli/deploy).
