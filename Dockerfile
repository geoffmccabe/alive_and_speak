# ═══════════════════════════════════════════════════════════════════
#  Hallo – RunPod Serverless Image
#  https://github.com/fudan-generative-vision/hallo
#
#  Fixes applied vs previous builds:
#    1. pip install -e /app   → installs the 'hallo' package so
#       "from hallo.animate..." resolves correctly
#    2. torch==2.2.2+cu121   → matches what the RunPod CUDA 11.8
#       runtime actually ships (logs showed 2.2.2+cu121)
#    3. xformers==0.0.25.post1 → built for torch 2.2.x + cu121
#    4. scipy==1.11.4 pinned  → last version fully compatible with
#       numpy 1.x (scipy 1.12+ requires numpy 2.x internals)
#    5. Install order: numpy → torch → scipy → xformers → hallo pkg
#       → rest of requirements → runtime deps
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

# ── 4. Patch requirements.txt before any installs ─────────────────
RUN sed -i \
        -e 's/numpy[>=<!=].*/numpy<=1.26.4/g' \
        -e 's/scipy[>=<!=].*/scipy==1.11.4/g' \
        -e 's/xformers[>=<!=].*/xformers==0.0.25.post1/g' \
        -e 's/onnxruntime-gpu[>=<!=].*/onnxruntime-gpu==1.16.3/g' \
        -e 's/insightface[>=<!=].*/insightface==0.7.3/g' \
        -e 's/diffusers[>=<!=].*/diffusers==0.27.2/g' \
        -e 's/transformers[>=<!=].*/transformers==4.38.2/g' \
        /app/requirements.txt

# ── 5. Install in strict dependency order ─────────────────────────
# numpy first so every subsequent C extension builds against 1.x ABI
RUN pip install "numpy<=1.26.4"

# torch 2.2.2+cu121 — matches what the RunPod host runtime exposes
# (logs showed: PyTorch 2.2.2+cu121 already present at runtime)
RUN pip install \
        torch==2.2.2 \
        torchvision==0.17.2 \
        torchaudio==2.2.2 \
        --index-url https://download.pytorch.org/whl/cu121

# scipy pinned BEFORE xformers so its C exts link against numpy 1.x
RUN pip install "scipy==1.11.4"

# xformers built for torch 2.2.x + cu121
RUN pip install "xformers==0.0.25.post1"

# Other binary deps
RUN pip install \
        "onnxruntime-gpu==1.16.3" \
        "insightface==0.7.3"

# ── 6. Install hallo as a Python package ──────────────────────────
# THIS is the critical fix for "ModuleNotFoundError: No module named 'hallo'"
# The repo ships a setup.py / pyproject.toml; editable install makes
# `from hallo.animate.face_animate import ...` resolve correctly.
RUN pip install -e /app --no-deps

# ── 7. Install remaining requirements (skip already-pinned pkgs) ──
RUN pip install --no-deps -r /app/requirements.txt

# ── 8. Runtime + handler deps ─────────────────────────────────────
RUN pip install \
        "diffusers==0.27.2" \
        "transformers==4.38.2" \
        "huggingface_hub<0.26.0" \
        "runpod==1.6.2" \
        requests accelerate pyyaml omegaconf einops \
        imageio imageio-ffmpeg face_alignment

# ── 9. Directory layout ───────────────────────────────────────────
RUN mkdir -p /tmp/hallo_outputs /runpod-volume/weights/hallo /app/configs

# ── 10. Application files ─────────────────────────────────────────
COPY handler.py             /app/handler.py
COPY start.sh               /start.sh
COPY extra_model_paths.yaml /app/configs/extra_model_paths.yaml

RUN chmod +x /start.sh

CMD ["/start.sh"]
