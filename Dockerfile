# ═══════════════════════════════════════════════════════════════════
#  MultiTalk (alive_and_speak) – RunPod Serverless Image
#  Base: CUDA 12.1.1 + cuDNN 8 + Ubuntu 22.04
# ═══════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ── Build-time defaults ──────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Conda
    CONDA_DIR=/opt/conda \
    CONDA_DEFAULT_ENV=multitalk \
    # flash-attn parallel compile jobs (lower if OOM during build)
    MAX_JOBS=4 \
    # Target GPU architectures (A100=8.0, A10=8.6, 4090=8.9, H100=9.0)
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Prepend the conda env bin first so every pip/python call uses it
ENV PATH="${CONDA_DIR}/envs/multitalk/bin:${CONDA_DIR}/bin:${PATH}"

# ── System packages ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git curl ca-certificates \
        ffmpeg \
        libsndfile1 libsndfile1-dev \
        libgl1-mesa-glx libglib2.0-0 \
        build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda ────────────────────────────────────────────────────
RUN wget -q \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda clean -ya

# ── Python 3.10 environment ──────────────────────────────────────
RUN conda create -n multitalk python=3.10 -y \
    && conda clean -ya

# ── PyTorch 2.4.1 + torchvision + torchaudio (CUDA 12.1) ────────
RUN pip install \
        torch==2.4.1 \
        torchvision==0.19.1 \
        torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121

# ── xformers (CUDA 12.1 build) ──────────────────────────────────
RUN pip install -U xformers==0.0.28 \
        --index-url https://download.pytorch.org/whl/cu121

# ── flash-attn pre-requisites ────────────────────────────────────
RUN pip install ninja psutil packaging "misaki[en]"

# ── flash-attn 2.7.4 (compiled from source, ~15 min) ────────────
# MAX_JOBS is set in ENV above; ninja parallelises the build.
RUN pip install flash_attn==2.7.4.post1 --no-build-isolation

# ── Clone alive_and_speak repo ───────────────────────────────────
WORKDIR /app
RUN git clone --depth 1 \
        https://github.com/saif816/alive_and_speak /app

# ── Repo Python dependencies ─────────────────────────────────────
RUN pip install -r /app/requirements.txt

# ── librosa via conda (avoids binary incompatibilities) ──────────
RUN conda install -c conda-forge librosa -y \
    && conda clean -ya

# ── RunPod SDK + HTTP client ─────────────────────────────────────
RUN pip install runpod requests

# ── Runtime directories ──────────────────────────────────────────
RUN mkdir -p /tmp/multitalk_outputs \
             /comfyui/ComfyUI

# ── Application files ────────────────────────────────────────────
COPY handler.py             /app/handler.py
COPY start.sh               /start.sh
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY extra_model_paths.yaml /comfyui/ComfyUI/extra_model_paths.yaml

RUN chmod +x /start.sh

CMD ["/start.sh"]
