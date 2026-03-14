#!/bin/bash
set -euo pipefail

# Setup script for the standalone WebSocket policy server on RunPod.
# Run from the openpi repo root after cloning and checking out the branch.
#
# Usage: bash setup_ws_server.sh

cd ~/openpi

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

apt-get update
apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev

pip install -U "huggingface_hub[cli,hf_transfer]"

uv pip install fastapi uvicorn websockets

TRANSFORMERS_DIR=$(python -c "import transformers; print(transformers.__file__.rsplit('/', 1)[0])")
echo "Copying custom transformers to: $TRANSFORMERS_DIR"
cp -r src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR/"
find "$TRANSFORMERS_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
python -c "
import sys
sys.modules.pop('transformers', None)
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly(), 'Custom transformers not installed correctly'
"
echo "Custom transformers verified successfully"

# Disable torch.compile for faster cold starts
sed -i 's/self\.sample_actions = torch\.compile(self\.sample_actions, mode="max-autotune")/#self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")/' \
    src/openpi/models_pytorch/pi0_pytorch.py 2>/dev/null || true

echo "WS server setup complete."
