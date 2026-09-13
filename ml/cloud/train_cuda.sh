#!/usr/bin/env bash
# Run a fresh, single-GPU control experiment from this source checkout.
# Provisioning, dependency installation and dataset downloads are explicit
# setup steps in ml/CLOUD_TRAINING.md; this script performs none of them.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash ml/cloud/train_cuda.sh DATA_ROOT NEW_OUTPUT_DIR [EPOCHS=12]"
  echo "Use an active CUDA Python environment, or set PYTHON_BIN to its Python executable."
  exit 0
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash ml/cloud/train_cuda.sh DATA_ROOT NEW_OUTPUT_DIR [EPOCHS=12]" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-python}"
epochs="${3:-12}"
if [[ ! "$epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer." >&2
  exit 2
fi
command -v "$python_bin" >/dev/null
# Resolve supplied paths before changing directories; paths with spaces work.
data_root="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$1")"
output_dir="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$2")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ ! -d "$data_root" ]]; then
  echo "DATA_ROOT is missing. Download or transfer Endoscapes first; see ml/CLOUD_TRAINING.md." >&2
  exit 2
fi
if [[ -e "$output_dir" ]]; then
  echo "Output already exists; choose a fresh directory to preserve prior experiments." >&2
  exit 2
fi
cd "$repo_root"

# An explicit CUDA failure must never turn into an unnoticed CPU or Mac run.
"$python_bin" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required.")
import torch
import torchvision
from holospex_ml.model import build_model, freeze_batchnorm, resolve_device
device = resolve_device("cuda")
model = build_model(num_classes=7, pretrained=False).to(device).train()
freeze_batchnorm(model)
# Exercise this architecture's forward/backward kernels before data preparation.
output = model(torch.randn(1, 3, 64, 64, device=device))
(output["out"].square().mean() + 0.4 * output["aux"].square().mean()).backward()
torch.cuda.synchronize()
print(f"CUDA preflight passed: {torch.cuda.get_device_name(0)}")
print(f"torch={torch.__version__}, torchvision={torchvision.__version__}, CUDA={torch.version.cuda}")
PY

mkdir -p "$output_dir"
export TORCH_HOME="${TORCH_HOME:-$repo_root/ml/weights}"
"$python_bin" -m pip freeze > "$output_dir/requirements-runtime.txt"
"$python_bin" -m holospex_ml doctor > "$output_dir/environment.json"

# Mac manifests contain absolute paths. Rebuild them from the unchanged source
# data here, preserving the established ignored-label and quarantine policies.
"$python_bin" -m holospex_ml prepare-endoscapes \
  --data-root "$data_root" --fps 25 --ignore-source-id 255 \
  --exclude-frame 153_32700 --output "$output_dir/manifest.json" \
  2>&1 | tee "$output_dir/prepare.log"

# First cloud run keeps the selected local recipe for a fair hardware control.
# This is fresh fine-tuning from generic weights, not resumption of best.pt.
"$python_bin" -m holospex_ml train \
  --manifest "$output_dir/manifest.json" --output-dir "$output_dir/train" \
  --device cuda --epochs "$epochs" --batch-size 2 --seed 42 \
  --width 672 --height 384 --lr 0.0003 --class-weighting balanced \
  2>&1 | tee "$output_dir/train.log"

# Compare on the original annotation grid. Further tuning must stay on val.
"$python_bin" -m holospex_ml evaluate-original \
  --manifest "$output_dir/manifest.json" --checkpoint "$output_dir/train/best.pt" \
  --split val --device cuda --output "$output_dir/train/metrics-val-original.json" \
  2>&1 | tee "$output_dir/evaluate.log"

echo "Finished. Save the complete output directory before ending the GPU runtime: $output_dir"
