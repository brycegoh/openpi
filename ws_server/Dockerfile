# Dockerfile for standalone WebSocket policy server.
#
# Build:
#   docker build -t openpi-ws-server -f ws_server/Dockerfile .
#
# Run (local):
#   docker run --rm -it --gpus=all -p 8000:8000 \
#       -e HF_TOKEN=$HF_TOKEN \
#       openpi-ws-server
#
# The build context must be the repository root so that pyproject.toml,
# uv.lock, and source files are available.

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /tmp/openpi

# System dependencies (git-lfs for LeRobot, ffmpeg 7 for av/PyAV)
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    linux-headers-generic \
    build-essential \
    clang \
    pkg-config \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg7 \
    && apt-get update \
    && apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavdevice-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV PATH="/.venv/bin:$PATH"

RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

# Clone openpi repository (same commit as modal image)
RUN git clone https://github.com/brycegoh/openpi.git .
RUN git checkout 37667384ec82975fdd23e700590f09de97b68eed

# Install openpi and its dependencies
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync
RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Apply custom transformers (AdaRMS support for pi0.5 models)
RUN TRANSFORMERS_DIR=$(/.venv/bin/python -c "import transformers; print(transformers.__file__.rsplit('/', 1)[0])") && \
    echo "Copying custom transformers to: $TRANSFORMERS_DIR" && \
    cp -r src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR/" && \
    find "$TRANSFORMERS_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    echo "Verifying custom transformers installation..." && \
    /.venv/bin/python -c "import sys; sys.modules.pop('transformers', None); from transformers.models.siglip import check; assert check.check_whether_transformers_replace_is_installed_correctly(), 'Custom transformers not installed correctly'" && \
    echo "Custom transformers verified successfully"

# Disable torch.compile for faster cold starts
RUN sed -i 's/self\.sample_actions = torch\.compile(self\.sample_actions, mode="max-autotune")/#self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")/' \
    /tmp/openpi/src/openpi/models_pytorch/pi0_pytorch.py

# Install server dependencies
RUN uv pip install fastapi uvicorn websockets huggingface_hub hf_transfer

ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Create checkpoint directory
RUN mkdir -p /inference-checkpoints
ENV CHECKPOINT_BASE_DIR=/inference-checkpoints

# Copy the standalone server into the image
COPY ws_server/serve_policy.py /tmp/openpi/ws_server/serve_policy.py

EXPOSE 8000

CMD ["python", "/tmp/openpi/ws_server/serve_policy.py", "--host", "0.0.0.0", "--port", "8000"]
