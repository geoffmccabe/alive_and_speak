# ═══════════════════════════════════════════════════════════════════
#  FLOAT – RunPod Serverless Image
#  https://github.com/deepbrainai-research/float
#
#  Base: CUDA 11.8 + Python 3.8.5
#  Models expected on network volume at /runpod-volume/weights/float/
# ═══════════════════════════════════════════════════════════════════
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CONDA_DIR=/opt/conda \
    CONDA_DEFAULT_ENV=float

ENV PATH="${CONDA_DIR}/envs/float/bin:${CONDA_DIR}/bin:${PATH}"

# ── System packages ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git curl ca-certificates \
        ffmpeg \
        libgl1-mesa-glx libglib2.0-0 \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda ────────────────────────────────────────────────────
RUN wget -q \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda clean -ya

# ── Python 3.8.5 environment ─────────────────────────────────────
RUN conda create -n float python=3.8.5 -y && conda clean -ya

# ── PyTorch 2.0.1 + CUDA 11.8 ────────────────────────────────────
RUN pip install \
        torch==2.0.1 \
        torchvision==0.15.2 \
        torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118

# ── Clone FLOAT repo ─────────────────────────────────────────────
WORKDIR /app
RUN git clone --depth 1 \
        https://github.com/deepbrainai-research/float /app

# ── FLOAT requirements ────────────────────────────────────────────
RUN pip install -r /app/requirements.txt

# ── gdown for Google Drive model download ────────────────────────
RUN pip install gdown

# ── RunPod SDK + requests ────────────────────────────────────────
RUN pip install runpod requests

# ── Runtime directories ──────────────────────────────────────────
RUN mkdir -p /tmp/float_outputs \
             /comfyui/ComfyUI

# ── Application files ────────────────────────────────────────────
COPY handler.py             /app/handler.py
COPY start.sh               /start.sh
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY extra_model_paths.yaml /comfyui/ComfyUI/extra_model_paths.yaml

RUN chmod +x /start.sh

CMD ["/start.sh"]
