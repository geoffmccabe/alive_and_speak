# ═══════════════════════════════════════════════════════════════════
# Hallo – Optimized Build Profile (Anti-Loop Unified Layering)
# ═══════════════════════════════════════════════════════════════════
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive 
ENV PYTHONUNBUFFERED=1 
ENV PIP_NO_CACHE_DIR=1

# ── 1. Install All System Toolchains in One Go ─────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget git curl ca-certificates ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsndfile1 \
    build-essential libasound2-dev portaudio19-dev \
    python3.10 python3.10-dev python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Configure Python Envs ──────────────────────────────────────
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && pip install --upgrade pip setuptools wheel

# ── 3. Initialize App Base Repository ──────────────────────────────
WORKDIR /app
RUN git clone --depth 1 https://github.com/fudan-generative-vision/hallo.git /app

# ── 4. Unified Dependency Injection (Eliminates Retries & Loops) ───
# We combine PyTorch, fundamental drivers, and requirements overrides into ONE step.
RUN sed -i 's/numpy==.*/numpy/g' /app/requirements.txt && \
    sed -i 's/onnxruntime-gpu==.*/onnxruntime-gpu==1.16.3/g' /app/requirements.txt && \
    sed -i 's/insightface==.*/insightface==0.7.3/g' /app/requirements.txt && \
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install onnxruntime-gpu==1.16.3 insightface==0.7.3 && \
    pip install --ignore-installed -r /app/requirements.txt && \
    pip install "runpod==1.6.2" requests accelerate transformers diffusers pyyaml

# ── 5. System Layout and Context Copying ───────────────────────────
RUN mkdir -p /tmp/hallo_outputs /runpod-volume/weights/hallo /app/configs

COPY handler.py             /app/handler.py
COPY start.sh               /start.sh
COPY extra_model_paths.yaml /app/configs/extra_model_paths.yaml
RUN chmod +x /start.sh

CMD ["/start.sh"]
