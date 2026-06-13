FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# System
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common wget git curl ca-certificates ffmpeg \
        libgl1-mesa-glx libglib2.0-0 libsndfile1 \
        build-essential libasound2-dev portaudio19-dev \
        python3.10 python3.10-dev python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10 + pip
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Clone Hallo3
RUN git clone --depth 1 https://github.com/fudan-generative-vision/hallo3.git /app

# Fix invalid dependency name in requirements
# Repo uses av==12.1.0; PyPI package name is "av", not "pyav"
RUN sed -i 's/pyav==14\.0\.1/av==12.1.0/g' /app/requirements.txt

# Install project requirements
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# RunPod + runtime helpers
RUN python -m pip install --no-cache-dir runpod==1.6.2 requests

# Directories
RUN mkdir -p /tmp/hallo3_outputs /workspace/weights/hallo

# App files
COPY handler.py /app/handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
