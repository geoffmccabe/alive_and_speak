# ═══════════════════════════════════════════════════════════════════
#  FLOAT – RunPod Serverless — CPU-only build
#  No CUDA required. Runs on cheap CPU workers.
#  Cost: ~$0.001–0.003 per generation vs ~$0.04 on GPU.
#  Speed: ~2–4 min per video on CPU vs ~35 sec on GPU.
# ═══════════════════════════════════════════════════════════════════
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System packages + Python 3.8 ─────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3.8-distutils \
        wget git curl ca-certificates \
        ffmpeg \
        libgl1-mesa-glx libglib2.0-0 \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.8 as default + pip ──────────────────────────────────
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && wget -q https://bootstrap.pypa.io/pip/3.8/get-pip.py -O /tmp/get-pip.py \
    && python3.8 /tmp/get-pip.py \
    && rm /tmp/get-pip.py

# ── PyTorch CPU-only (much smaller, no CUDA) ──────────────────────
RUN pip install \
        torch==2.0.1 \
        torchvision==0.15.2 \
        torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cpu

# ── Clone your FLOAT fork (with CPU generate.py changes) ─────────
WORKDIR /app
RUN git clone --depth 1 \
        https://github.com/saif816/float /app

# ── Copy the CPU-patched generate.py over the original ───────────
COPY generate.py /app/generate.py

# ── All Python dependencies ───────────────────────────────────────
RUN pip install -r /app/requirements.txt
RUN pip install \
        "huggingface_hub[cli]" \
        gdown \
        edge-tts \
        "runpod==1.6.2" \
        requests

# ── Download all models into image ───────────────────────────────
RUN mkdir -p /app/checkpoints
COPY download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py

# float.pth from Google Drive
RUN gdown "1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0" \
        -O /app/checkpoints/float.pth

# ── Runtime directories ──────────────────────────────────────────
RUN mkdir -p /tmp/float_outputs /comfyui/ComfyUI

# ── Application files ────────────────────────────────────────────
COPY handler.py             /app/handler.py
COPY start.sh               /start.sh
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY extra_model_paths.yaml /comfyui/ComfyUI/extra_model_paths.yaml

RUN chmod +x /start.sh

CMD ["/start.sh"]
