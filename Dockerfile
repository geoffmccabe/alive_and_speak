# ═══════════════════════════════════════════════════════════════════
#  FLOAT – RunPod Serverless Image
#  All models baked in — no volume needed for inference.
#  edge-tts included: text + voice → audio → talking portrait video
# ═══════════════════════════════════════════════════════════════════
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System packages + Python 3.8 via deadsnakes PPA ──────────────
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

# ── Python 3.8 as default + pip (must use 3.8-specific URL) ──────
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && wget -q https://bootstrap.pypa.io/pip/3.8/get-pip.py -O /tmp/get-pip.py \
    && python3.8 /tmp/get-pip.py \
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

# ── All Python dependencies ───────────────────────────────────────
RUN pip install -r /app/requirements.txt
RUN pip install \
        "huggingface_hub[cli]" \
        gdown \
        edge-tts \
        "runpod==1.6.2" \
        requests

# ─────────────────────────────────────────────────────────────────
#  Download all models into /app/checkpoints/
#
#  1. float.pth                                        ~789 MB  Google Drive
#  2. wav2vec2-base-960h/                              ~360 MB  HuggingFace
#       config.json, model.safetensors, preprocessor_config.json,
#       tokenizer_config.json, vocab.json, special_tokens_map.json
#  3. wav2vec-english-speech-emotion-recognition/      ~360 MB  HuggingFace
#       config.json, pytorch_model.bin, preprocessor_config.json
# ─────────────────────────────────────────────────────────────────
RUN mkdir -p /app/checkpoints

# 1. float.pth — main FLOAT model
RUN gdown "1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0" \
        -O /app/checkpoints/float.pth

# 2. wav2vec2-base-960h — audio feature encoder (all files)
RUN python - << 'PYEOF'
from huggingface_hub import hf_hub_download
import os

repo  = "facebook/wav2vec2-base-960h"
dest  = "/app/checkpoints/wav2vec2-base-960h"
files = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
]
os.makedirs(dest, exist_ok=True)
for f in files:
    try:
        path = hf_hub_download(repo_id=repo, filename=f, local_dir=dest)
        print(f"  ✓  {f}")
    except Exception as e:
        print(f"  ⚠  {f}: {e}")
PYEOF

# 3. wav2vec-english-speech-emotion-recognition — emotion model (all files)
RUN python - << 'PYEOF'
from huggingface_hub import hf_hub_download
import os

repo  = "r-f/wav2vec-english-speech-emotion-recognition"
dest  = "/app/checkpoints/wav2vec-english-speech-emotion-recognition"
files = [
    "config.json",
    "pytorch_model.bin",
    "preprocessor_config.json",
]
os.makedirs(dest, exist_ok=True)
for f in files:
    try:
        path = hf_hub_download(repo_id=repo, filename=f, local_dir=dest)
        print(f"  ✓  {f}")
    except Exception as e:
        print(f"  ⚠  {f}: {e}")
PYEOF

# ── Verify all files are present before finishing build ───────────
RUN python - << 'PYEOF'
import os, sys

ckpts = "/app/checkpoints"
required = [
    "float.pth",
    "wav2vec2-base-960h/config.json",
    "wav2vec2-base-960h/model.safetensors",
    "wav2vec2-base-960h/preprocessor_config.json",
    "wav2vec2-base-960h/tokenizer_config.json",
    "wav2vec2-base-960h/vocab.json",
    "wav2vec2-base-960h/special_tokens_map.json",
    "wav2vec-english-speech-emotion-recognition/config.json",
    "wav2vec-english-speech-emotion-recognition/pytorch_model.bin",
    "wav2vec-english-speech-emotion-recognition/preprocessor_config.json",
]

missing = []
for f in required:
    path = os.path.join(ckpts, f)
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1_000_000
        print(f"  ✓  {f}  ({mb:.1f} MB)")
    else:
        print(f"  ✗  MISSING: {f}")
        missing.append(f)

if missing:
    print(f"\nBUILD FAILED: {len(missing)} required file(s) missing.")
    sys.exit(1)
else:
    print("\nAll model files verified — build complete.")
PYEOF

# ── Runtime directories ──────────────────────────────────────────
RUN mkdir -p /tmp/float_outputs /comfyui/ComfyUI

# ── Application files ────────────────────────────────────────────
COPY handler.py             /app/handler.py
COPY start.sh               /start.sh
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY extra_model_paths.yaml /comfyui/ComfyUI/extra_model_paths.yaml

RUN chmod +x /start.sh

CMD ["/start.sh"]
