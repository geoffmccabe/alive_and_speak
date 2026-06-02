FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CONDA_DIR=/opt/conda \
    CONDA_DEFAULT_ENV=multitalk \
    MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

ENV PATH="${CONDA_DIR}/envs/multitalk/bin:${CONDA_DIR}/bin:${PATH}"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git curl ca-certificates \
        ffmpeg \
        libsndfile1 libsndfile1-dev \
        libgl1-mesa-glx libglib2.0-0 \
        build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda clean -ya \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Python 3.10 environment
RUN conda create -n multitalk python=3.10 pip -y \
    && conda clean -ya

# Use the env explicitly for all Python installs
SHELL ["conda", "run", "-n", "multitalk", "/bin/bash", "-c"]

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.4.1 + torchvision + torchaudio (CUDA 12.1)
RUN python -m pip install \
        torch==2.4.1 \
        torchvision==0.19.1 \
        torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121

# xformers (allow PyPI fallback, keep CUDA wheel index available)
RUN python -m pip install xformers==0.0.28 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# flash-attn pre-requisites
RUN python -m pip install ninja psutil packaging "misaki[en]"

# flash-attn 2.7.4 (compiled from source)
RUN python -m pip install flash_attn==2.7.4.post1 --no-build-isolation

# Clone repo
WORKDIR /app
RUN git clone --depth 1 \
        https://github.com/saif816/alive_and_speak /app

# Repo Python dependencies
RUN python -m pip install -r /app/requirements.txt

# librosa via conda-forge
RUN conda install -c conda-forge librosa -y \
    && conda clean -ya

# RunPod SDK + HTTP client
RUN python -m pip install runpod requests

# Runtime directories
RUN mkdir -p /tmp/multitalk_outputs \
             /comfyui/ComfyUI

# Application files
COPY handler.py             /app/handler.py
COPY start.sh               /start.sh
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY extra_model_paths.yaml /comfyui/ComfyUI/extra_model_paths.yaml

RUN chmod +x /start.sh

CMD ["/start.sh"]
