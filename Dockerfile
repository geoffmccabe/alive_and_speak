FROM python:3.8-slim-bullseye
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES="" \
    TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git curl ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install \
        torch==2.0.1+cpu \
        torchvision==0.15.2+cpu \
        torchaudio==2.0.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
# Force fresh clone
ARG CACHE_BUST=20260624
RUN git clone --depth 1 https://github.com/saif816/float /app
# ── Replace FMT.py with corrected version (self.opt = opt in right place) ──
COPY FMT.py /app/models/float/FMT.py





# ------------------------------
# PATCH FLOAT AUTOMATICALLY
# ------------------------------
# Replace opt.device -> cpu
# Inject opt.device after parse_args()
RUN python - <<'PY'
from pathlib import Path
f = Path("/app/generate.py")
txt = f.read_text()
old = "opt = parser.parse_args()"
new = """
opt = parser.parse_args()
import torch
if not hasattr(opt, 'device'):
    opt.device = torch.device('cpu')
"""
if old in txt:
    txt = txt.replace(old, new)
f.write_text(txt)
print("generate.py patched")
PY
# Verify patch
RUN grep -n "device" /app/generate.py || true
RUN pip install -r /app/requirements.txt
RUN pip install \
        "huggingface_hub[cli]" \
        gdown \
        edge-tts \
        runpod==1.6.2 \
        requests \
        librosa \
        soundfile \
        pydub
RUN mkdir -p /app/checkpoints
COPY download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py
RUN gdown "1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0" \
        -O /app/checkpoints/float.pth
RUN mkdir -p /tmp/float_outputs /comfyui/ComfyUI
COPY handler.py /app/handler.py
COPY start.sh /start.sh
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY extra_model_paths.yaml /comfyui/ComfyUI/extra_model_paths.yaml
RUN chmod +x /start.sh
CMD ["/start.sh"]
