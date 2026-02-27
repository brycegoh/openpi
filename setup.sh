#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash setup.sh <dataset/repo-name>"
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN is not set in the environment."
  exit 1
fi

export DATASET_REPO_ID="$1"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export HF_HOME="/workspace/hf-cache"
export OPENPI_DATA_HOME="/workspace/openpi-cache"

cd /workspace/openpi
pip install uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
apt update
apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev tmux
pip install -U "huggingface_hub[cli,hf_transfer]"
python download_dataset_and_base_model.py
uv pip show transformers
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/ 