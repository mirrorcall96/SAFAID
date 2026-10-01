#!/bin/bash
# One-time setup of the GPU machine: pinned Python environment (the exact versions of the paper runs) and the
# base model at a pinned revision. Data are fetched separately with safaid/fetch_data.py.
#
# Usage:   bash scripts/setup.sh
# Env:     SAFAID_WORKDIR   working directory for data/, runs/, logs/   (default /workspace/safaid)
#          PYTHON           interpreter to install into                (default: python on PATH)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKDIR=${SAFAID_WORKDIR:-/workspace/safaid}
export PATH=/venv/main/bin:/opt/instance-tools/bin:$PATH   # vast.ai PyTorch image (venv + instance tools); harmless elsewhere
mkdir -p "$WORKDIR" && cd "$WORKDIR"
mkdir -p logs runs
PY=${PYTHON:-$(command -v python)}
echo "python: $PY"; "$PY" -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

# keep the image's torch build (2.9.1+cu128 in vastai/pytorch:2.9.1-cuda-12.8.1-py312-24.04); pin everything else
TORCH_VER=$("$PY" -c "import torch; print(torch.__version__)")
CONSTRAINTS=$(mktemp)
echo "torch==$TORCH_VER" > "$CONSTRAINTS"
"$PY" -m pip install -q -c "$CONSTRAINTS" -r "$REPO/requirements.txt"
rm -f "$CONSTRAINTS"
"$PY" -m pip freeze > logs/pip_freeze.txt
"$PY" -c "import torch, transformers, trl, unsloth, albumentations; print('OK', torch.__version__, transformers.__version__, trl.__version__, unsloth.__version__, albumentations.__version__)"

export HF_HUB_ENABLE_HF_TRANSFER=1
"$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('unsloth/Qwen3-VL-4B-Instruct', revision='252d592b59b0233b226875a44ac135cfa1d3f755'))"

df -h "$WORKDIR" | tail -1
echo SETUP_DONE | tee logs/SETUP_DONE
