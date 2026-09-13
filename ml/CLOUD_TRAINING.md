# Training Holospex on a Google GPU

The existing PyTorch model supports NVIDIA CUDA. The completed local runs used
Apple MPS; this handoff prepares **new cloud experiments**. No cloud resource or
training job has been launched. Project, billing/credits, region, and GPU access
still need to be established before provisioning a paid runtime.

For this two-day hack, use one GPU and the existing CLI. A Google Cloud notebook
or terminal with a current CUDA-enabled PyTorch environment is sufficient.
Google's [PyTorch VM guide](https://docs.cloud.google.com/deep-learning-vm/docs/pytorch_start_instance)
covers project billing, GPU quota, drivers, and access; verify the selected image
meets our package requirements rather than copying its older example versions.
If Cloud setup delays the hack, a hosted Colab GPU notebook is the quickest
fallback to try. GPU availability and session lifetime are not guaranteed.
[Colab FAQ](https://research.google.com/colaboratory/faq.html)

## Move the current source

Upload `ml/outputs/cloud-handoff/holospex-cloud-source.tar.gz` to the runtime and
extract it into a fresh directory:

```sh
tar -xzf holospex-cloud-source.tar.gz
cd holospex
```

The archive has a top-level `holospex/` directory and contains the current ML
source, tests, configuration, cloud runner, documentation, and shared contracts.
It excludes datasets, checkpoints, credentials, and the Mac virtual environment.
Use this working-tree bundle: several implemented ML files are still untracked,
so cloning the remote repository or archiving Git HEAD would omit them.

Keep `contracts/` beside `ml/`. Export validation reads the canonical schemas
from that checkout, and diagnostic images use the shared anatomy catalog. Do
not copy only the installed Python package or create a second class map.

## Use the GPU runtime's Python

Use Python 3.11 or later and a compatible CUDA-enabled PyTorch/Torchvision pair
meeting `ml/pyproject.toml` (`torch>=2.8,<3`, `torchvision>=0.23,<1`). Preserve a
working runtime pair when installing the remaining dependencies. If the pair is
too old, select a compatible runtime or install a matched pair using the official
[PyTorch installation instructions](https://pytorch.org/get-started/locally/).
`ml/requirements-macos.txt` records the Mac environment; it is not a CUDA lockfile.

From the extracted repository root, using the runtime's active Python:

```sh
python -m pip install -e './ml[train]'
python -m holospex_ml doctor
python -m unittest discover -s ml/tests -v
```

`doctor` must report CUDA available. The runner explicitly requests CUDA and
fails when it is unavailable; it does not silently train on the CPU. In a
notebook, run shell blocks in a `%%bash` cell and `cd` to the extracted repository
inside that cell. This runs on the hosted runtime, not a Colab local connection.

## Prepare data and run a matched cloud control

The existing selected dataset occupies about 107 MB. Either upload the existing
`ml/data/endoscapes/` directory with its original LICENSE, README, and download
provenance, or explicitly retrieve the same author release:

```sh
python -m holospex_ml download-endoscapes --output-dir ml/data/endoscapes
bash ml/cloud/train_cuda.sh ml/data/endoscapes ml/outputs/cloud-cuda-001 12
```

The downloader fetches only the labeled pairs and metadata; it does not require
the full 6.29 GB ZIP. Data terms and source attribution are in
[TRAINING.md](TRAINING.md). The runner itself does not download the dataset.

The runner accepts `DATA_ROOT OUTPUT_DIR [EPOCHS]`, defaulting to 12 epochs. Use a
fresh output directory for each experiment. Set `PYTHON_BIN` to the runtime's
Python executable if its name is not `python`. It checks CUDA, regenerates the
manifest inside the output directory, trains into `OUTPUT_DIR/train`, then
evaluates on the original-resolution **validation** labels.

Regeneration is necessary because the Mac manifest contains absolute image/mask
paths. It preserves the explicit source-255 ignore policy, exclusion of frame
`153_32700`, original video splits, and 25-fps filename assumption. Path changes
produce a different manifest digest; they do not change the intended samples.

The first cloud control uses the selected local recipe: 672 × 384 input, batch
size 2, learning rate 0.0003, balanced class weighting, seed 42, and generic
pretrained initialization. Retain these settings for the migration check;
cross-device results need not be bitwise identical. Compare original-resolution
validation metrics with [SMALL_ANATOMY.md](SMALL_ANATOMY.md). After the control,
change one setting at a time in separate runs. Do not tune on the test split or
the unlabeled public demonstration clip.

## Keep results when the GPU session ends

Copy the whole experiment directory to durable storage or download it before
ending the runtime. Preserve the manifest, environment record, configuration,
history, validation reports, `best.pt`, and `last.pt` together. Copy the source
bundle too so another teammate can reproduce the exact code used. Stop the GPU
runtime when the work finishes; provision and retain storage deliberately.

Checkpoints store CPU tensors and can be evaluated or used for inference on
CUDA, MPS, or CPU. They currently do **not** store optimizer/RNG state, and the
training CLI has no resume or warm-start option. This runner starts a fresh
experiment; it cannot continue the old run exactly after an interruption. A
future warm-start from the selected weights would be a separate experiment
with a new optimizer. Existing frame contracts and the frontend handoff remain
the same when a cloud-trained checkpoint is selected.
