# Dockerfile for v2.1 dataset processing with OpenPI norm stats computation.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app/openpi

# Install system dependencies (git-lfs for LeRobot, ffmpeg 7 for av/PyAV)
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    linux-headers-generic \
    build-essential \
    clang \
    pkg-config \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add ffmpeg 7 PPA (required by av>=14.0.0)
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

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Add venv to PATH so Modal can find Python
ENV PATH="/.venv/bin:$PATH"

# Create the virtual environment
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

# Clone openpi repository (feat/rtc branch from fork)
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --branch feat/rtc --single-branch --recurse-submodules https://github.com/brycegoh/openpi.git .

# Install openpi and its dependencies using the lockfile
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync
RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Install additional packages needed for serving
RUN uv pip install fastapi websockets msgpack-numpy sortedcontainers

RUN cp -r src/openpi/models_pytorch/transformers_replace/* /.venv/lib/python3.11/site-packages/transformers/ && \
    find /.venv/lib/python3.11/site-packages/transformers -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Set HF transfer for faster downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1
