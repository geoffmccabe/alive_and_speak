# ═══════════════════════════════════════════════════════════════════
#  Hallo – RunPod Serverless Image
#  https://github.com/fudan-generative-vision/hallo
#
#  ROOT CAUSE FIX: scipy 1.15+ requires NumPy 2.x internals.
#  We pin scipy<=1.13.0 and xformers==0.0.23 (built for torch 2.1.2+cu118)
#  so that the scipy → xformers → diffusers import chain never breaks.
# ═══════════════════════════════════════════════════════════════════
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

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
    && pip install --upgrade pip setuptools wheel

# ── 3. Clone Hallo repo ───────────────────────────────────────────
WORKDIR /app
RUN git clone --depth 1 https://github.com/fudan-generative-vision/hallo.git /app

# ── 4. Patch requirements.txt BEFORE installing anything ─────────
#
#  Key pins that fix the crash:
#    numpy<=1.26.4      – stay on 1.x ABI
#    scipy==1.11.4      – last scipy that works cleanly with numpy 1.x
#                         (scipy 1.12+ introduced the _multiufuncs issue)
#    xformers==0.0.23   – built against torch 2.1.x + cu118, numpy 1.x
#    diffusers==0.27.2  – version Hallo's scripts were written for
#    transformers==4.38.2
#    huggingface_hub<0.26.0 – avoids cached_download removal
#
RUN sed -i \
        -e 's/numpy[>=<!=].*/numpy<=1.26.4/g' \
        -e 's/scipy[>=<!=].*/scipy==1.11.4/g' \
        -e 's/xformers[>=<!=].*/xformers==0.0.23/g' \
        -e 's/onnxruntime-gpu[>=<!=].*/onnxruntime-gpu==1.16.3/g' \
        -e 's/insightface[>=<!=].*/insightface==0.7.3/g' \
        -e 's/diffusers[>=<!=].*/diffusers==0.27.2/g' \
        -e 's/transformers[>=<!=].*/transformers==4.38.2/g' \
        /app/requirements.txt

# ── 5. Install in correct order ───────────────────────────────────
#  Order matters: numpy first, then scipy (so its C extensions
#  compile/link against numpy 1.x), then torch, then everything else.
RUN pip install "numpy<=1.26.4"

RUN pip install \
        torch==2.1.2 \
        torchvision==0.16.2 \
        torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu118

# scipy must be pinned BEFORE xformers is installed
RUN pip install "scipy==1.11.4"

RUN pip install \
        "xformers==0.0.23" \
        "onnxruntime-gpu==1.16.3" \
        "insightface==0.7.3"

# Install the rest of Hallo's requirements (numpy/scipy/xformers already satisfied)
RUN pip install --no-deps -r /app/requirements.txt

# Runtime + handler deps
RUN pip install \
        "diffusers==0.27.2" \
        "transformers==4.38.2" \
        "huggingface_hub<0.26.0" \
        "runpod==1.6.2" \
        requests accelerate pyyaml omegaconf einops imageio imageio-ffmpeg \
        face_alignment

# ── 6. Directory layout ───────────────────────────────────────────
RUN mkdir -p /tmp/hallo_outputs /runpod-volume/weights/hallo /app/configs

# ── 7. Application files ──────────────────────────────────────────
COPY handler.py             /app/handler.py
COPY start.sh               /start.sh
COPY extra_model_paths.yaml /app/configs/extra_model_paths.yaml

RUN chmod +x /start.sh

CMD ["/start.sh"]
