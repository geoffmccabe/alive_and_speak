#!/bin/bash
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Hallo – RunPod Serverless Worker                        ║"
echo "╚══════════════════════════════════════════════════════════╝"

HALLO_WEIGHTS="${HALLO_WEIGHTS:-/runpod-volume/weights/hallo}"
echo "  HALLO_WEIGHTS : ${HALLO_WEIGHTS}"
echo ""

# /app/pretrained_models → volume (relative paths in default.yaml)
if [[ ! -L "/app/pretrained_models" ]]; then
    [[ -d "/app/pretrained_models" ]] && mv /app/pretrained_models /app/pretrained_models.bak
    ln -sf "${HALLO_WEIGHTS}" /app/pretrained_models
    echo "  ✓ Linked /app/pretrained_models → ${HALLO_WEIGHTS}"
else
    echo "  ✓ /app/pretrained_models already linked"
fi

# .cache dir — Hallo's tensor_to_video hardcodes ".cache/output.mp4" relative to cwd=/app
mkdir -p /app/.cache
echo "  ✓ /app/.cache ready"

echo ""
echo "  Checking model files:"
for f in \
    "hallo/net.pth" \
    "motion_module/mm_sd_v15_v2.ckpt" \
    "sd-vae-ft-mse/config.json" \
    "stable-diffusion-v1-5/unet/config.json" \
    "face_analysis/models/det_10g.onnx" \
    "face_analysis/models/buffalo_l/det_10g.onnx" \
    "wav2vec/wav2vec2-base-960h/config.json" \
    "audio_separator/Kim_Vocal_2.onnx"
do
    full="${HALLO_WEIGHTS}/${f}"
    if [[ -f "${full}" || -L "${full}" ]]; then
        echo "    ✓  ${f}"
    else
        echo "    ✗  MISSING: ${f}"
    fi
done

echo ""
echo "  Starting handler…"
exec python3 -u /app/handler.py
