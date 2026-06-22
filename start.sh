#!/bin/bash
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  FLOAT – RunPod Serverless Worker                        ║"
echo "╚══════════════════════════════════════════════════════════╝"

FLOAT_WEIGHTS="${FLOAT_WEIGHTS:-/runpod-volume/weights/float}"
APP_CKPTS="/app/checkpoints"

echo ""

if [[ -d "${FLOAT_WEIGHTS}" ]]; then
    echo "  Volume detected — overriding baked-in checkpoints"
    if [[ -d "${APP_CKPTS}" && ! -L "${APP_CKPTS}" ]]; then
        mv "${APP_CKPTS}" "${APP_CKPTS}_baked"
    fi
    if [[ ! -L "${APP_CKPTS}" ]]; then
        ln -sf "${FLOAT_WEIGHTS}" "${APP_CKPTS}"
        echo "  Linked: ${APP_CKPTS} → ${FLOAT_WEIGHTS}"
    else
        echo "  Already linked: ${APP_CKPTS}"
    fi
else
    echo "  No volume override — using baked-in checkpoints"
fi

echo ""
echo "  Checking model files:"

MISSING=0
check() {
    local f="${APP_CKPTS}/$1"
    if [[ -f "${f}" ]]; then
        sz=$(du -sh "${f}" | cut -f1)
        echo "    ✓  $1  (${sz})"
    else
        echo "    ✗  MISSING: $1"
        MISSING=$((MISSING + 1))
    fi
}

# float main model
check "float.pth"

# wav2vec2 audio encoder — both weight formats required
check "wav2vec2-base-960h/config.json"
check "wav2vec2-base-960h/model.safetensors"
check "wav2vec2-base-960h/preprocessor_config.json"
check "wav2vec2-base-960h/tokenizer_config.json"
check "wav2vec2-base-960h/vocab.json"
check "wav2vec2-base-960h/special_tokens_map.json"

# emotion recognition model
check "wav2vec-english-speech-emotion-recognition/config.json"
check "wav2vec-english-speech-emotion-recognition/pytorch_model.bin"
check "wav2vec-english-speech-emotion-recognition/preprocessor_config.json"

echo ""
if [[ ${MISSING} -gt 0 ]]; then
    echo "  ⚠  ${MISSING} file(s) missing — worker may fail on first job."
else
    echo "  ✓ All model files verified."
fi

echo ""
echo "  Starting RunPod handler…"
echo ""

exec python -u /app/handler.py
