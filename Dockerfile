# ═══════════════════════════════════════════════════════════════════
# Hallo – RunPod Serverless Image (Optimized for 16GB-24GB VRAM)
# ═══════════════════════════════════════════════════════════════════
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive 
ENV PYTHONUNBUFFERED=1 
ENV PIP_NO_CACHE_DIR=1

# ── System packages + Python 3.10 ──────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget git curl ca-certificates \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 \
    libsndfile1 \
    build-essential \
    python3.10 python3.10-dev python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# ── Make python3.10 the default python ────────────────────────────
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ── Install Pip ──────────────────────────────────────────────────
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# ── PyTorch 2.1.2 + CUDA 11.8 ────────────────────────────────────
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# ── Clone Hallo repo ─────────────────────────────────────────────
WORKDIR /app
RUN git clone --depth 1 https://github.com/fudan-generative-vision/hallo.git /app

# ── Hallo requirements ────────────────────────────────────────────
RUN pip install -r /app/requirements.txt
RUN pip install "runpod==1.6.2" requests choice-of-packages accelerate transformers diffusers

# ── Runtime directories ──────────────────────────────────────────
RUN mkdir -p /tmp/hallo_outputs /runpod-volume/weights/hallo

# ── Application files ────────────────────────────────────────────
COPY handler.py             /app/handler.py
COPY start.sh               /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
