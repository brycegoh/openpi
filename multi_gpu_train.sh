#!/bin/bash
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN is not set in the environment."
  exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "Error: WANDB_API_KEY is not set in the environment."
  exit 1
fi

uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_pytorch.py pi05_tcr_full_finetune_pytorch \
--exp_name pi05_B006_B016 \
--checkpoint-base-dir /workspace/checkpoints \
--data.repo-id /workspace/dataset \
--pytorch_weight_path /workspace/base_checkpoints/pi05_base_pytorch \
--wandb-enabled \
--batch-size 256