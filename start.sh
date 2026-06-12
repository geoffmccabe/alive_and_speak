#!/bin/bash
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Hallo3 – RunPod Serverless Worker                       ║"
echo "╚══════════════════════════════════════════════════════════╝"

HALLO3_WEIGHTS="${HALLO3_WEIGHTS:-/workspace/weights/hallo3/pretrained_models}"
echo "  HALLO3_WEIGHTS : ${HALLO3_WEIGHTS}"
echo ""

# /app/pretrained_models → weights volume (Hallo3 uses relative paths)
if [[ ! -L "/app/pretrained_models" ]]; then
    [[ -d "/app/pretrained_models" ]] && mv /app/pretrained_models /app/pretrained_models.bak
    ln -sf "${HALLO3_WEIGHTS}" /app/pretrained_models
    echo "  ✓ Linked /app/pretrained_models → ${HALLO3_WEIGHTS}"
else
    echo "  ✓ /app/pretrained_models linked"
fi

echo ""
echo "  Checking model files:"
for f in \
    "hallo3/1/mp_rank_00_model_states.pt" \
    "hallo3/latest" \
    "cogvideox-5b-i2v-sat/transformer/1/mp_rank_00_model_states.pt" \
    "cogvideox-5b-i2v-sat/vae/3d-vae.pt" \
    "t5-v1_1-xxl/config.json" \
    "wav2vec/wav2vec2-base-960h/config.json" \
    "audio_separator/Kim_Vocal_2.onnx" \
    "face_analysis/models/scrfd_10g_bnkps.onnx"
do
    full="${HALLO3_WEIGHTS}/${f}"
    if [[ -f "${full}" || -L "${full}" ]]; then
        echo "    ✓  ${f}"
    else
        echo "    ✗  MISSING: ${f}"
    fi
done

echo ""
echo "  Starting handler…"
exec python3 -u /app/handler.py
