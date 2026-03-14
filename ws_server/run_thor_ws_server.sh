#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-openpi-ws-server-thor}"
HOST_PORT="${HOST_PORT:-8000}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/openpi}"
CONTAINER_CACHE_DIR="/root/.cache/openpi"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

usage() {
  cat <<EOF
Usage: bash ws_server/run_thor_ws_server.sh [options]

Run the Thor-based OpenPI WebSocket server container with a persistent cache.

Options:
  --image-name NAME         Docker image tag to run
                            (default: $IMAGE_NAME)
  --host-port PORT          Host port to publish
                            (default: $HOST_PORT)
  --container-port PORT     Container port to publish
                            (default: $CONTAINER_PORT)
  --cache-dir PATH          Host cache directory to mount
                            (default: $CACHE_DIR)
  --env-file PATH           Optional env file to source for HF_TOKEN
                            (default: $ENV_FILE)
  --skip-env-file           Do not source an env file
  -h, --help                Show this help message

Examples:
  bash ws_server/run_thor_ws_server.sh
  bash ws_server/run_thor_ws_server.sh --host-port 9000
  HF_TOKEN=... bash ws_server/run_thor_ws_server.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-name)
      IMAGE_NAME="${2:?missing value for --image-name}"
      shift 2
      ;;
    --host-port)
      HOST_PORT="${2:?missing value for --host-port}"
      shift 2
      ;;
    --container-port)
      CONTAINER_PORT="${2:?missing value for --container-port}"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="${2:?missing value for --cache-dir}"
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

if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Warning: HF_TOKEN is not set." >&2
  echo "Public checkpoints may still work, but private Hugging Face repos will fail." >&2
fi

mkdir -p "$CACHE_DIR"

DOCKER_CMD=(docker)
if ! docker info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1; then
    DOCKER_CMD=(sudo docker)
  else
    echo "Error: Docker is not available for the current user." >&2
    echo "Make sure Docker is installed and the daemon is running." >&2
    exit 1
  fi
fi

echo "Running image '$IMAGE_NAME' on port $HOST_PORT"
echo "Persistent cache: $CACHE_DIR"

"${DOCKER_CMD[@]}" run --rm -it --runtime nvidia \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -v "$CACHE_DIR:$CONTAINER_CACHE_DIR" \
  "$IMAGE_NAME"
