# Training Holospex on a Google GPU

The Vertex training workflow has completed five A100 experiments in
`us-central1`, project `eastwest72hack26bos-501`. All five jobs reached
`JOB_STATE_SUCCEEDED`, completed original-grid validation, and had their final
artifacts downloaded with checksum verification on **2026-09-13**.
See [CLOUD_EXPERIMENTS.md](CLOUD_EXPERIMENTS.md) for the frozen plan and job IDs,
and [CLOUD_RESULTS.md](CLOUD_RESULTS.md) for outcomes and model selection.

The subsequent six-job, three-seed loss comparison has also completed with all
jobs successful and artifacts verified. See [DICE_ITERATION.md](DICE_ITERATION.md),
[DICE_RESULTS.md](DICE_RESULTS.md) and its separate
[launch manifest](outputs/dice-handoff/launch-manifest.json). It reused the data
archive with a new source archive and kept 672 × 384 input. The CE + GDL variant
was not promoted. The [data catalog](DATA_CATALOG.json) records newly acquired
datasets in the private bucket for future experiments.

Two matched higher-resolution jobs also completed successfully: 896 × 512 job
`6125326098055036928` and 1120 × 640 job `3324087129830588416`. All 13 artifacts
per job were checksum-verified after collection. See
[RESOLUTION_ITERATION.md](RESOLUTION_ITERATION.md) for the frozen comparison,
validation tradeoffs, local CPU latency check, and model-selection decision.

The private bucket is `gs://eastwest72hack26bos-501-holospex-ml`.
The ignored [launch manifest](outputs/cloud-handoff/launch-manifest.json) records
both source/data bundle hashes and URIs, complete file inventories, and the five
filled job configurations. Preserve this record with the run results. Local
experiments previously ran on Apple MPS; these jobs explicitly require CUDA.

## Connection and workload identity

The local Google Cloud CLI is authenticated in configuration `holospex`, with
billing, Vertex AI, and Cloud Storage enabled for the hackathon project:

```sh
gcloud --configuration=holospex config list --format='yaml(core.account,core.project)'
```

Credentials stay in the CLI's normal local store. Do not copy API keys, login
files, or service-account JSON into source bundles, job configuration, or Git.
Inside Vertex, the bootstrap and artifact runner use Application Default
Credentials from the default **Custom Code Service Agent**:
`service-PROJECT_NUMBER@gcp-sa-aiplatform-cc.iam.gserviceaccount.com`.
The same-project bucket uses that workload identity's storage access. It differs
from the Vertex service agent that pulls container images. An optional custom
service account needs the submitter's `iam.serviceAccounts.actAs` permission
and the necessary bucket object permissions. See Google's
[service-account guide](https://docs.cloud.google.com/vertex-ai/docs/general/custom-service-account).

The connection check found quota for eight preemptible A100s in `us-central1`;
[the saved quota response](outputs/gcp-vertex-training-quotas.json) is a quota
snapshot, not a capacity reservation. Each configured job requests one A100.

## Container, bundles, and execution

[vertex-job.template.yaml](cloud/vertex-job.template.yaml) describes one
`a2-highgpu-1g` worker with one `NVIDIA_TESLA_A100` and a 100 GB SSD boot disk.
It uses the official PyTorch 2.8.0 / CUDA 12.6 runtime pinned to this image digest:

```text
docker.io/pytorch/pytorch@sha256:dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69
```

The manifest and container configuration were read directly from the public
registry and saved in [runtime verification](outputs/cloud-handoff/pytorch-runtime-verification.json).
Vertex [accepts Docker Hub images](https://docs.cloud.google.com/vertex-ai/docs/training/create-custom-container),
so this path needs no local image build or private image registry. The upstream
image defaults to Python 3.11; startup checks Python ≥3.11, Torch 2.8.0,
Torchvision 0.23.0, compatible imports, and an available CUDA device.

The job's `python -u -c` argument embeds [vertex_bootstrap.py](cloud/vertex_bootstrap.py).
Filled configurations are serialized as JSON, which the gcloud YAML config
reader also accepts. Do not manually concatenate shell code or credentials.
The bootstrap performs these steps:

1. Install the Cloud Storage SDK and download the two private GCS bundles.
2. Verify each complete bundle against its explicit SHA-256 before extraction.
3. Extract ordinary files/directories into fresh `/work/holospex` and
   `/work/endoscapes` roots, rejecting links and paths outside those roots.
4. Install `/work/holospex/ml[train]` with constraints that preserve Torch 2.8.0
   and Torchvision 0.23.0, then repeat the runtime checks.
5. Execute [vertex_entry.py](cloud/vertex_entry.py), which records the runtime,
   prepares the manifest, trains the model, and evaluates original-grid
   validation metrics while preserving artifacts in GCS.

The source bundle contains the working-tree ML code, tests, configuration,
documentation, shared `contracts/`, and explicit synthetic contract fixtures.
The separate data bundle contains the selected Endoscapes images, masks,
metadata, license, and download provenance. It reuses the acquired dataset;
startup does not retrieve new training data from the internet. Generic
Torchvision pretrained model weights may download on first use.

Keep source and data bundle objects immutable under their hash-named prefixes.
A code or data change needs a new bundle/hash and a new filled job configuration.
Do not replace the bundle behind an existing run. Keep `contracts/` next to
`ml/`, because validation and diagnostics read canonical files from that checkout.

Environment variables are `HOLOSPEX_PROJECT`, `HOLOSPEX_SOURCE_URI`,
`HOLOSPEX_SOURCE_SHA256`, `HOLOSPEX_DATA_URI`, `HOLOSPEX_DATA_SHA256`,
`HOLOSPEX_OUTPUT_URI`, `HOLOSPEX_RUN_NAME`, and `HOLOSPEX_EPOCHS`.
Optional experiment settings are `HOLOSPEX_ARCHITECTURE`,
`HOLOSPEX_AUGMENTATION`, `HOLOSPEX_SAMPLING`, `HOLOSPEX_SEED`,
`HOLOSPEX_LOSS` (`ce` or `ce_generalized_dice`), and `HOLOSPEX_DICE_WEIGHT`.
`HOLOSPEX_WIDTH` and `HOLOSPEX_HEIGHT` set the recorded training/evaluation
input grid; omitting them preserves 672 × 384.
Omitting the loss settings preserves the original balanced-CE workflow. The
matched Dice iteration is defined in [DICE_ITERATION.md](DICE_ITERATION.md).
The completed width/height comparison is recorded in
[RESOLUTION_ITERATION.md](RESOLUTION_ITERATION.md).
Their supported values and comparison rules are in
[CLOUD_EXPERIMENTS.md](CLOUD_EXPERIMENTS.md).

## Submit, inspect, and stop jobs

The first control has already been submitted. The following is the submission
command pattern for a new run; do not rerun it merely to check existing status:

```sh
gcloud --configuration=holospex ai custom-jobs create \
  --project=eastwest72hack26bos-501 --region=us-central1 \
  --display-name=NEW_RUN_NAME --config=PATH_TO_FILLED_JOB_CONFIG.json
```

The config contains a `CustomJobSpec` directly, without a `jobSpec` wrapper.
The [launch manifest](outputs/cloud-handoff/launch-manifest.json) maps all five
planned experiments to their generated config files and output prefixes.
Inspect the initial control with:

```sh
gcloud --configuration=holospex ai custom-jobs describe 3017512501681061888 \
  --project=eastwest72hack26bos-501 --region=us-central1 \
  --format='yaml(name,state,startTime,endTime,error)'
gcloud --configuration=holospex ai custom-jobs stream-logs 3017512501681061888 \
  --project=eastwest72hack26bos-501 --region=us-central1
```

If `stream-logs` is quiet, query Cloud Logging directly. This job's observed
startup messages use resource type `ml_job`; filtering for
`aiplatform_custom_job` would miss them. Match the job ID without fixing the
resource type, and bound the query to recent entries:

```sh
gcloud --configuration=holospex logging read \
  'resource.labels.job_id="3017512501681061888"' \
  --project=eastwest72hack26bos-501 --freshness=30m --limit=30 --order=desc \
  --format='table(timestamp,textPayload)'
```

Initial service logs reported provisioning for first-time framework usage.
Those messages indicate startup progress; wait for the bootstrap's runtime
check and training epoch logs before reporting that model training has started.

The job also appears in the
[Vertex console](https://console.cloud.google.com/vertex-ai/training/custom-jobs/3017512501681061888?project=eastwest72hack26bos-501&region=us-central1).
The configuration uses `scheduling.strategy: SPOT`, `timeout: 14400s`, and
`disableRetries: true`. The four-hour timeout bounds **running time**; it does
not bound how long a Spot job can wait for capacity. `maxWaitDuration` applies
to Flex Start rather than this Spot configuration. Check pending jobs and cancel
a job that cannot obtain capacity before switching resources:

```sh
gcloud --configuration=holospex ai custom-jobs cancel JOB_ID \
  --project=eastwest72hack26bos-501 --region=us-central1
```

Spot workers can be interrupted. Artifact preservation does not implement exact
resumption: current checkpoints omit optimizer, RNG, and sampler state, and the
CLI has no resume or warm-start option. A new invocation starts a fresh run in a
unique attempt directory. Never combine histories across attempts. Vertex
releases the worker resources when the job reaches a terminal state; the bucket
objects remain. See [Spot training](https://docs.cloud.google.com/vertex-ai/docs/training/use-spot-vms).

## Retrieve complete, consistent results

The artifact runner checks for completed epoch snapshots every 15 seconds and
after each subprocess finishes. It verifies that checkpoint epochs,
configuration, history, and selected validation metrics agree before uploading
a coherent epoch record. Snapshots are immutable and organized as:

```text
runs/RUN_NAME/attempts/ATTEMPT_ID/
  status/running.json
  status/completed.json                 # only after training and evaluation finish
  status/failed.json                    # when failure status can be uploaded
  epochs/0001.json                      # available completed-epoch snapshots
  objects/SHA256/filename               # immutable artifacts referenced by manifests
```

A hard preemption may leave only prior snapshots and `running.json`; it does not
guarantee a final failure marker. A completion record contains logical artifact
names mapped to object URIs, SHA-256 digests, and byte sizes. It includes
`train/best.pt`, `train/last.pt`, training history/configuration,
`train/metrics-val-original.json`, the manifest, logs, and runtime records.
Do not choose the newest-looking checkpoint object independently of its manifest.

List available completion records for the first control:

```sh
gcloud --configuration=holospex storage ls \
  'gs://eastwest72hack26bos-501-holospex-ml/runs/cloud-001-control/attempts/*/status/completed.json'
```

After selecting a completed attempt, collect it into a new local directory:

```sh
python ml/cloud/collect_vertex_results.py \
  --completion-uri 'gs://eastwest72hack26bos-501-holospex-ml/runs/cloud-001-control/attempts/ATTEMPT_ID/status/completed.json' \
  --output-dir ml/outputs/cloud-001-control
```

The helper uses the authenticated `holospex` gcloud configuration (override with
`--configuration`). It restores logical artifact paths such as `train/best.pt`,
verifies every SHA-256 digest and byte count, and publishes the directory only
after all downloads pass. It refuses existing output directories and incomplete
attempts. `cloud-completion.json`, `cloud-last-epoch.json` when referenced, and
`cloud-collection.json` retain the completion/epoch records and collection receipt.
Keep the original launch manifest alongside these files. Cloud metadata and
absolute dataset paths remain unchanged; this helper does not rewrite training
manifests. An interrupted attempt's epoch manifest can recover its coherent saved
outputs separately, but must remain labeled incomplete.

The cloud manifest regenerates absolute paths while preserving the original
case splits, source-255 ignore policy, quarantined frame `153_32700`, and 25-fps
filename assumption. The frozen experiments train on 343 frames and compare
on the same 75 original-resolution validation frames. They do not evaluate the
test split or use the public demo clip to select settings. The artifact files
can be downloaded for CUDA, MPS, or CPU inference; the segmentation wire
contracts stay unchanged.

## Reviewed partial training package

The [reviewed-data comparison](REVIEWED_DATA_RESULTS.md) completed six fresh paired
A100 jobs with nearly equal optimizer-update budgets. All three paired seeds
improved both small-anatomy pooled and equal-case IoU, with remaining plate
precision problems. All 93 artifacts passed checksum verification. Its
[launch manifest](outputs/reviewed-handoff/launch-manifest.json) records separate
source, original-data and prepared-pilot archive hashes. It reuses the original
base data archive and introduces no new dataset download.

Set both `HOLOSPEX_PREPARED_URI` and `HOLOSPEX_PREPARED_SHA256` for the expanded
arm. Bootstrap verifies and extracts this third archive into `/work/prepared`
and passes `--prepared-package /work/prepared/package.json` to the runner.
Omit both variables for the ordinary original-data workflow.

The prepared archive preserves exact original `manifest.json`,
`base-manifest.json`, review/resolution snapshots, derived semantic PNGs,
per-image provenance, summary and licenses. `package.json` has format version
`1.0.0`, artifact type `holospex_prepared_cloud_package`, both manifest SHA-256s,
and a file inventory: original absolute `sourcePath`, `root` (`data` or
`prepared`), relative `path`, `sha256` and `bytes`. Base files resolve beneath
`/work/endoscapes`; additional files resolve beneath `/work/prepared`.

Before training, the runner verifies every file binding, the exact original
sample prefix, new-case separation, class/ignore policy, semantic PNG labels,
dimensions and counts. It rebases operational paths into a separate run manifest;
it does not rewrite the original snapshots. Expanded runs preserve
`prepared-input-verification.json`, `prepared-package.json`,
`manifest-source.json` and `base-manifest-source.json`. Both arms preserve
`training-input-audit.json`, including ordered split-content fingerprints built
from sample metadata and actual image/mask SHA-256 hashes. This makes unchanged
validation/test inputs verifiable across the paired runs.

The frozen training source passed 192 ML tests after extraction. Training and
collection use the existing immutable epoch/artifact workflow. The analysis
helper is maintained separately from the frozen training source so finishing a
report cannot change already submitted training code.

## Manual CUDA fallback

A separately configured Google Cloud GPU notebook/VM or hosted Colab session
can still run the existing CLI. Google's [PyTorch VM guide](https://docs.cloud.google.com/deep-learning-vm/docs/pytorch_start_instance)
covers that route; [Colab GPU availability and session duration vary](https://research.google.com/colaboratory/faq.html).
Use a hosted GPU runtime and Python ≥3.11 with a compatible CUDA Torch/Torchvision
pair, rather than the Mac virtual environment or its dependency record.

Transfer and extract the same source/data bundles into a fresh directory,
preserving top-level `holospex/` and `endoscapes/`. From the source root:

```sh
python -m pip install -e './ml[train]'
python -m holospex_ml doctor
python -m unittest discover -s ml/tests -v
bash ml/cloud/train_cuda.sh ../endoscapes ml/outputs/cloud-manual-001 12
```

`doctor` and the runner must report CUDA availability. `train_cuda.sh` accepts
`DATA_ROOT NEW_OUTPUT_DIR [EPOCHS=12]`, regenerates the manifest, trains the
matched 672 × 384 / batch-2 / LR-0.0003 / seed-42 balanced-loss control, and
evaluates original-resolution validation labels. Set `PYTHON_BIN` if needed.
This manual runner does not install dependencies, download data, or synchronize
artifacts automatically. Save its entire output directory and bundle provenance
to durable storage before stopping the notebook/VM. Dataset terms remain in
[TRAINING.md](TRAINING.md).
