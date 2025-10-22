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

# Ensure CUDA compat libs are visible
RUN ldconfig /usr/local/cuda-12.1/compat/ || true

# Copy requirements first for better caching
COPY requirements.txt ./requirements.txt

# Python deps
RUN python3 -m pip install --no-cache-dir --upgrade pip


# Install MinerU
RUN python3 -m pip install 'mineru[core]' --break-system-packages \
 && python3 -m pip cache purge
 
RUN python3 -m pip install --no-cache-dir -r ./requirements.txt
# Create runtime dirs BEFORE downloading models
RUN mkdir -p /root/.mineru /root/.cache/huggingface ./models ./files_test ./output_minerU ./logs

# Download models with proper environment and config
# This is the critical fix - ensure models are downloaded to correct location
RUN export HOME=/root && \
    export MINERU_MODEL_SOURCE=local && \
    mkdir -p /root/.mineru && \
    echo '{"models": {}}' > /root/.mineru/magic-pdf.json && \
    mineru-models-download -s huggingface -m all && \
    ls -la /root/.mineru/ && \
    cat /root/.mineru/magic-pdf.json || echo "Config file missing"

# Copy app code
COPY . .

# Ensure correct permissions
RUN chmod -R 755 /root/.mineru ./models

# Set working directory
WORKDIR /

EXPOSE 5001

# Simple, reliable startup
CMD ["python3", "app.py"]