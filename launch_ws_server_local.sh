#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

HOST="0.0.0.0"
PORT="8888"
CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR:-$REPO_ROOT/inference-checkpoints}"
ENV_FILE="$REPO_ROOT/.env"
INSTALL_DEPS=0

usage() {
  cat <<EOF
Usage: bash launch_ws_server_local.sh [options]

Launch the standalone local WebSocket server from this repository.

Options:
  --host HOST              Bind host (default: 0.0.0.0)
  --port PORT              Bind port (default: 8000)
  --checkpoint-dir PATH    Directory for downloaded checkpoints
                           (default: $REPO_ROOT/inference-checkpoints)
  --env-file PATH          Load environment variables from a .env file
                           (default: $REPO_ROOT/.env)
  --skip-env-file          Do not load a .env file
  --install-deps           Run uv dependency setup before launch
  -h, --help               Show this help message

Examples:
  bash launch_ws_server_local.sh
  bash launch_ws_server_local.sh --port 8080
  bash launch_ws_server_local.sh --checkpoint-dir /tmp/openpi-checkpoints
  bash launch_ws_server_local.sh --install-deps
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:?missing value for --host}"
      shift 2
      ;;
    --port)
      PORT="${2:?missing value for --port}"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="${2:?missing value for --checkpoint-dir}"
      shift 2
      ;;
    --env-file)
      ENV_FILE="${2:?missing value for --env-file}"
      shift 2
      ;;
    --skip-env-file)
      ENV_FILE=""
      shift
      ;;
    --install-deps)
      INSTALL_DEPS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is not installed or not on PATH." >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ -n "$ENV_FILE" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  else
    echo "Warning: env file '$ENV_FILE' not found; continuing without it." >&2
  fi
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$REPO_ROOT/.cache/openpi}"
export CHECKPOINT_BASE_DIR="$CHECKPOINT_DIR"

mkdir -p "$HF_HOME" "$OPENPI_DATA_HOME" "$CHECKPOINT_BASE_DIR"

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  echo "Installing local server dependencies with uv..."
  GIT_LFS_SKIP_SMUDGE=1 uv sync
  GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
  uv pip install fastapi uvicorn websockets "huggingface_hub[cli,hf_transfer]"
fi

if ! uv run python -c "import fastapi, uvicorn, websockets, huggingface_hub" >/dev/null 2>&1; then
  echo "Error: local server dependencies are missing." >&2
  echo "Run this script with --install-deps, or install them manually with uv." >&2
  exit 1
fi

if ! uv run python -c "import openpi" >/dev/null 2>&1; then
  echo "Error: the openpi package is not installed in the uv environment." >&2
  echo "Run 'GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .' first," >&2
  echo "or rerun this script with --install-deps." >&2
  exit 1
fi

if ! uv run python -c "import sys; sys.modules.pop('transformers', None); from transformers.models.siglip import check; raise SystemExit(0 if check.check_whether_transformers_replace_is_installed_correctly() else 1)" >/dev/null 2>&1; then
  TRANSFORMERS_DIR="$(uv run python -c "import transformers; print(transformers.__file__.rsplit('/', 1)[0])")"
  echo "Copying custom transformers to: $TRANSFORMERS_DIR"
  cp -r src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR/"
  find "$TRANSFORMERS_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
fi

uv run python -c "import sys; sys.modules.pop('transformers', None); from transformers.models.siglip import check; assert check.check_whether_transformers_replace_is_installed_correctly(), 'Custom transformers not installed correctly'"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Warning: HF_TOKEN is not set. Public checkpoints may still work, but private Hugging Face repos will fail to download." >&2
fi

echo "Launching local WebSocket server on ${HOST}:${PORT}"
echo "Checkpoint cache: $CHECKPOINT_BASE_DIR"

exec uv run python ws_server/serve_policy.py \
  --host "$HOST" \
  --port "$PORT" \
  --checkpoint-dir "$CHECKPOINT_BASE_DIR"
