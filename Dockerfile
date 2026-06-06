# ═══════════════════════════════════════════════════════════════════
#  Hallo – RunPod Serverless Image
#  https://github.com/fudan-generative-vision/hallo
# ═══════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── 1. System packages ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        wget git curl ca-certificates ffmpeg \
        libgl1-mesa-glx libglib2.0-0 libsndfile1 \
        build-essential libasound2-dev portaudio19-dev \
        python3.10 python3.10-dev python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Python base ────────────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && python3.10 -m pip install --upgrade pip setuptools wheel

# ── 3. Clone Hallo repo ───────────────────────────────────────────
WORKDIR /app
RUN git clone --depth 1 https://github.com/fudan-generative-vision/hallo.git /app

# ── 4. Patch requirements.txt before any installs ────────────────
RUN sed -i \
        -e 's/^numpy[>=<!=].*/numpy<=1.26.4/g' \
        -e 's/^scipy[>=<!=].*/scipy==1.11.4/g' \
        -e 's/^xformers[>=<!=].*/xformers==0.0.25.post1/g' \
        -e 's/^onnxruntime-gpu[>=<!=].*/onnxruntime-gpu==1.16.3/g' \
        -e 's/^insightface[>=<!=].*/insightface==0.7.3/g' \
        -e 's/^diffusers[>=<!=].*/diffusers==0.27.2/g' \
        -e 's/^transformers[>=<!=].*/transformers==4.38.2/g' \
        -e 's/^protobuf[>=<!=].*/protobuf==3.20.3/g' \
        /app/requirements.txt

# ── 5. Core deps in strict order ──────────────────────────────────
RUN python3.10 -m pip install "numpy<=1.26.4"

RUN python3.10 -m pip install \
        torch==2.2.2 \
        torchvision==0.17.2 \
        torchaudio==2.2.2 \
        --index-url https://download.pytorch.org/whl/cu121

RUN python3.10 -m pip install "scipy==1.11.4"
RUN python3.10 -m pip install "xformers==0.0.25.post1"
RUN python3.10 -m pip install "protobuf==3.20.3"

RUN python3.10 -m pip install \
        "onnxruntime-gpu==1.16.3" \
        "insightface==0.7.3"

# ── 6. Install hallo as a Python package ──────────────────────────
RUN python3.10 -m pip install -e /app --no-deps

# ── 7. Install remaining requirements ─────────────────────────────
RUN python3.10 -m pip install -r /app/requirements.txt

# ── 8. Explicit moviepy stack + proglog ───────────────────────────
RUN python3.10 -m pip install \
        "moviepy==1.0.3" \
        "proglog>=0.1.9" \
        "decorator>=4.0.2"

# ── 9. Runtime + handler deps ─────────────────────────────────────
RUN python3.10 -m pip install \
        "diffusers==0.27.2" \
        "transformers==4.38.2" \
        "huggingface_hub<0.26.0" \
        "runpod==1.6.2" \
        requests accelerate pyyaml omegaconf einops \
        imageio imageio-ffmpeg face_alignment

# ── 10. Directory layout ──────────────────────────────────────────
RUN mkdir -p /tmp/hallo_outputs /runpod-volume/weights/hallo /app/configs

# ── 11. Application files ─────────────────────────────────────────
COPY handler.py             /app/handler.py
COPY start.sh               /start.sh
COPY extra_model_paths.yaml /app/configs/extra_model_paths.yaml

RUN chmod +x /start.sh

CMD ["/start.sh"]
