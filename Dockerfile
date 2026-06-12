# ═══════════════════════════════════════════════════════════════════
#  Hallo3 – RunPod Serverless Image
#  https://github.com/fudan-generative-vision/hallo3
#  CVPR 2025 — CogVideoX-5B DiT backbone
# ═══════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# ── 1. System packages ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common wget git curl ca-certificates ffmpeg \
        libgl1-mesa-glx libglib2.0-0 libsndfile1 \
        build-essential libasound2-dev portaudio19-dev \
        python3.10 python3.10-dev python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Python ─────────────────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && pip install --upgrade pip setuptools wheel

# ── 3. Clone Hallo3 ───────────────────────────────────────────────
WORKDIR /app
RUN git clone --depth 1 https://github.com/fudan-generative-vision/hallo3.git /app

# ── 4. PyTorch 2.4.0 + cu121 ─────────────────────────────────────
RUN pip install \
        torch==2.4.0 \
        torchvision==0.19.0 \
        torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

# ── 5. Core deps ──────────────────────────────────────────────────
RUN pip install \
        "numpy==1.26.4" \
        "deepspeed==0.14.4" \
        "SwissArmyTransformer==0.4.12" \
        "omegaconf==2.3.0" \
        "einops==0.8.0" \
        "transformers==4.45.2" \
        "diffusers" \
        "accelerate" \
        "safetensors==0.4.3" \
        "sentencepiece==0.2.0" \
        "tokenizers==0.20.1"

# ── 6. Media / face analysis deps ────────────────────────────────
RUN pip install \
        "insightface==0.7.3" \
        "onnxruntime-gpu==1.19.2" \
        "mediapipe==0.10.14" \
        "opencv-python==4.10.0.84" \
        "imageio==2.34.2" \
        "imageio-ffmpeg==0.5.1" \
        "moviepy==1.0.3" \
        "proglog==0.1.10" \
        "decorator>=4.0.2" \
        "librosa==0.10.2.post1" \
        "audio-separator==0.21.2" \
        "decord==0.6.0"

# ── 7. Install requirements (no editable install — Hallo3 has no setup.py)
RUN pip install -r /app/requirements.txt

# ── 8. RunPod + requests ──────────────────────────────────────────
RUN pip install "runpod==1.6.2" requests

# ── 9. Directories ────────────────────────────────────────────────
RUN mkdir -p /tmp/hallo3_outputs /workspace/weights/hallo3

# ── 10. Application files ─────────────────────────────────────────
COPY handler.py /app/handler.py
COPY start.sh   /start.sh

RUN chmod +x /start.sh
CMD ["/start.sh"]
