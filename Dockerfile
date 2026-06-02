FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_DIR=/opt/venv \
    MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

ENV PATH="${VENV_DIR}/bin:${PATH}"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip \
        wget \
        git \
        curl \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        build-essential \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Virtual environment
RUN python3.10 -m venv "${VENV_DIR}" \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Copy app code
COPY . /app

# PyTorch CUDA 12.1
RUN python -m pip install \
        torch==2.4.1 \
        torchvision==0.19.1 \
        torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121

# xformers
RUN python -m pip install xformers==0.0.28 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Core build/runtime deps
RUN python -m pip install \
        ninja \
        psutil \
        packaging \
        "misaki[en]" \
        runpod \
        requests \
        librosa

# flash-attn
RUN python -m pip install flash_attn==2.7.4.post1 --no-build-isolation

# Repo dependencies
RUN python -m pip install -r /app/requirements.txt

# Runtime directories
RUN mkdir -p /tmp/multitalk_outputs \
             /comfyui/ComfyUI

# Extra model path config
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY extra_model_paths.yaml /comfyui/ComfyUI/extra_model_paths.yaml

# Start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
