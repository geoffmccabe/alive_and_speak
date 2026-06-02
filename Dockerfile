FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
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

# Create virtual env
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Clone alive_and_speak repo
WORKDIR /app
RUN git clone --depth 1 https://github.com/saif816/alive_and_speak /app

# Upgrade to PyTorch 2.5.1 (Fixes the infer_schema type-hint validation bug)
RUN pip install \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Matching xformers version for PyTorch 2.5.1
RUN pip install xformers==0.0.28.post3 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Core deps
RUN pip install \
        ninja \
        psutil \
        packaging \
        "misaki[en]" \
        runpod \
        requests \
        librosa

# Repo Python dependencies (Installed before flash-attn to prevent overwriting versions)
RUN pip install -r /app/requirements.txt

# flash-attn compiled against the upgraded PyTorch stack
RUN pip install flash_attn==2.7.4.post1 --no-build-isolation

# Runtime directories
RUN mkdir -p /tmp/multitalk_outputs \
             /comfyui/ComfyUI

# App files
COPY handler.py /app/handler.py
COPY start.sh /start.sh
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY extra_model_paths.yaml /comfyui/ComfyUI/extra_model_paths.yaml

RUN chmod +x /start.sh

CMD ["/start.sh"]
