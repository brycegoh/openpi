#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-openpi-ws-server-thor}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-$REPO_ROOT/ws_server/Dockerfile.thor}"

usage() {
  cat <<EOF
Usage: bash ws_server/build_thor_ws_server.sh [options]

Build the Thor-based OpenPI WebSocket server image.

Options:
  --image-name NAME         Docker image tag to build
                            (default: $IMAGE_NAME)
  --dockerfile PATH         Dockerfile to build
                            (default: $DOCKERFILE_PATH)
  -h, --help                Show this help message

Examples:
  bash ws_server/build_thor_ws_server.sh
  bash ws_server/build_thor_ws_server.sh --image-name my-ws-server
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-name)
      IMAGE_NAME="${2:?missing value for --image-name}"
      shift 2
      ;;
    --dockerfile)
      DOCKERFILE_PATH="${2:?missing value for --dockerfile}"
      shift 2
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

echo "Building image '$IMAGE_NAME' from '$DOCKERFILE_PATH'"
"${DOCKER_CMD[@]}" build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" "$REPO_ROOT"
