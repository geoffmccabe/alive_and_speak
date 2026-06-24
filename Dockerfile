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

# Clone YOUR fork of FLOAT here, not the old upstream repo
RUN git clone --depth 1 \
        https://github.com/saif816/float /app

RUN pip install -r /app/requirements.txt
RUN pip install \
        "huggingface_hub[cli]" \
        gdown \
        edge-tts \
        "runpod==1.6.2" \
        requests

RUN mkdir -p /app/checkpoints
COPY download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py

RUN gdown "1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0" \
        -O /app/checkpoints/float.pth

RUN mkdir -p /tmp/float_outputs /comfyui/ComfyUI

# Override cloned files with your patched versions from this repo
COPY generate.py /app/generate.py
COPY handler.py /app/handler.py
COPY models/float/FMT.py /app/models/float/FMT.py
COPY models/float/FLOAT.py /app/models/float/FLOAT.py
COPY models/float/styledecoder.py /app/models/float/styledecoder.py

COPY start.sh /start.sh
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY extra_model_paths.yaml /comfyui/ComfyUI/extra_model_paths.yaml

RUN chmod +x /start.sh

CMD ["/start.sh"]
