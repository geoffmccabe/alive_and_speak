# ═══════════════════════════════════════════════════════════════════
#  FLOAT – RunPod Serverless Image
#  https://github.com/deepbrainai-research/float
#
#  Uses Python 3.8 via deadsnakes PPA — no conda needed.
#  Models live on network volume at /runpod-volume/weights/float/
# ═══════════════════════════════════════════════════════════════════
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System packages + Python 3.8 via deadsnakes PPA ─────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        python3-pip \
        wget git curl ca-certificates \
        ffmpeg \
        libgl1-mesa-glx libglib2.0-0 \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Make python3.8 the default python ────────────────────────────
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# ── pip for python3.8 ────────────────────────────────────────────
RUN wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
    && python /tmp/get-pip.py \
    && rm /tmp/get-pip.py

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

# ── gdown for Google Drive checkpoint download ───────────────────
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
