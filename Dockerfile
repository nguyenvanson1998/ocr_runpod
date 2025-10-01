# Dockerfile for MinerU API Service (fixed OpenCV deps & entrypoint)
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MINERU_MODEL_SOURCE=local

# System deps (fonts + OpenCV runtime + basics)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        ca-certificates \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        ffmpeg \
        libopencv-dev \
        python3-opencv \
        build-essential \
        curl \
        git \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# Ensure CUDA compat libs are visible (usually harmless if already present)
RUN ldconfig /usr/local/cuda-12.1/compat/ || true


# Copy requirements first for better caching
COPY requirements.txt ./requirements.txt

# Python deps
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir -r ./requirements.txt

# App code
COPY . .

# NOTE: If your base image already has pip in /usr/local, --break-system-packages helps on Ubuntu 22.04
# RUN python3 -m pip install 'mineru[core]' --break-system-packages \
#  && python3 -m pip cache purge

# (Optional) Pre-download models during build. Comment this out if you prefer to pull at runtime.
RUN /bin/bash -lc "mineru-models-download -s huggingface -m all || true"

# Create runtime dirs
RUN mkdir -p ./files_test ./output_minerU ./logs ./models

# Ensure we run as root (this is default, but explicit)
USER root

EXPOSE 5001

# Simple, reliable startup
CMD ["python3", "app.py"]





