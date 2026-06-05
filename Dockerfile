# ═══════════════════════════════════════════════════════════════════
# Hallo – RunPod Serverless Image (Fixed Blinker & Toolchain Build)
# ═══════════════════════════════════════════════════════════════════
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive 
ENV PYTHONUNBUFFERED=1 
ENV PIP_NO_CACHE_DIR=1

# ── System packages + Python 3.10 + Compilation Toolchains ────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget git curl ca-certificates \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 \
    libsndfile1 \
    build-essential \
    libasound2-dev portaudio19-dev \
    python3.10 python3.10-dev python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# ── Force Python 3.10 as default ──────────────────────────────────
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ── Upgrade foundational deployment build tools ───────────────────
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && pip install --upgrade pip setuptools wheel

# ── Install CUDA 11.8 Compatible PyTorch Ecosystem ────────────────
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# ── Pull the repository code ──────────────────────────────────────
WORKDIR /app
RUN git clone --depth 1 https://github.com/fudan-generative-vision/hallo.git /app

# ── Direct Wheel Installation for problematic components ──────────
RUN pip install onnxruntime-gpu==1.16.3
RUN pip install insightface==0.7.3

# ── Loosen strict version requirements to bypass pip loops ─────────
RUN sed -i 's/numpy==.*/numpy/g' /app/requirements.txt && \
    sed -i 's/onnxruntime-gpu==.*/onnxruntime-gpu==1.16.3/g' /app/requirements.txt && \
    sed -i 's/insightface==.*/insightface==0.7.3/g' /app/requirements.txt

# ── Install remaining requirements (Overriding legacy packages) ───
# The --ignore-installed flag prevents the "Cannot uninstall blinker" distutils error.
RUN pip install --ignore-installed -r /app/requirements.txt
RUN pip install "runpod==1.6.2" requests accelerate transformers diffusers pyyaml

# ── Target environment layout mapping ─────────────────────────────
RUN mkdir -p /tmp/hallo_outputs /runpod-volume/weights/hallo /app/configs

# ── Codebase Placement ────────────────────────────────────────────
COPY handler.py             /app/handler.py
COPY start.sh               /start.sh
COPY extra_model_paths.yaml /app/configs/extra_model_paths.yaml
RUN chmod +x /start.sh

CMD ["/start.sh"]
