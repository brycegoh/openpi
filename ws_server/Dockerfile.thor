# WebSocket server image layered on top of the prebuilt PI 0.5 runtime.
#
# Build from the repository root:
#   docker build -t openpi-ws-server-thor -f ws_server/Dockerfile.thor .
#
# Run:
#   docker run --rm -it --runtime nvidia \
#     -p 8000:8000 \
#     -e HF_TOKEN="$HF_TOKEN" \
#     -v "$HOME/.cache/openpi":/root/.cache/openpi \
#     openpi-ws-server-thor
#
# This image bakes in the manual container steps that were previously needed:
# install ws server deps, set PYTHONPATH, and start the standalone server.
# The mounted `/root/.cache/openpi` volume persists checkpoints, Hugging Face
# cache, and downloaded norm stats.

FROM brycegohgl/thor-pi:latest

WORKDIR /workspace

# Copy only the files needed to run the standalone WebSocket server.
COPY pyproject.toml uv.lock /workspace/
COPY src /workspace/src
COPY packages/openpi-client /workspace/packages/openpi-client
COPY ws_server /workspace/ws_server

# Add the lightweight server runtime dependencies on top of the base image.
RUN python -m pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    "huggingface_hub[cli]"

ENV PYTHONPATH=/workspace/packages/openpi-client/src:/workspace/src:/workspace
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/root/.cache/openpi/huggingface
ENV OPENPI_DATA_HOME=/root/.cache/openpi
ENV OPENPI_LOCAL_DATASET_DIR=/root/.cache/openpi/dataset
ENV CHECKPOINT_BASE_DIR=/root/.cache/openpi/inference-checkpoints

RUN mkdir -p \
    /root/.cache/openpi \
    /root/.cache/openpi/dataset \
    /root/.cache/openpi/huggingface \
    /root/.cache/openpi/inference-checkpoints

EXPOSE 8000

ENTRYPOINT ["python", "ws_server/serve_policy.py"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--checkpoint-dir", "/root/.cache/openpi/inference-checkpoints"]
