#!/bin/bash
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Hallo – RunPod Serverless Worker                        ║"
echo "╚══════════════════════════════════════════════════════════╝"

HALLO_WEIGHTS="${HALLO_WEIGHTS:-/runpod-volume/weights/hallo}"
echo "  HALLO_WEIGHTS : ${HALLO_WEIGHTS}"
echo ""

# ── InsightFace CWD fix ───────────────────────────────────────────
# inference.py runs with cwd=/app and InsightFace resolves paths as:
#   ./pretrained_models/face_analysis/models/*.onnx
# So we need /app/pretrained_models/face_analysis/models/ → volume models/
mkdir -p /app/pretrained_models/face_analysis
if [[ ! -L "/app/pretrained_models/face_analysis/models" ]]; then
    ln -sf "${HALLO_WEIGHTS}/face_analysis/models" \
           /app/pretrained_models/face_analysis/models
    echo "  ✓ Linked: /app/pretrained_models/face_analysis/models → volume"
else
    echo "  ✓ Already linked: /app/pretrained_models/face_analysis/models"
fi

# ── Model file check ──────────────────────────────────────────────
echo ""
echo "  Checking model files:"
for f in \
    "hallo/net.pth" \
    "motion_module/mm_sd_v15_v2.ckpt" \
    "sd-vae-ft-mse/config.json" \
    "stable-diffusion-v1-5/unet/config.json" \
    "face_analysis/models/det_10g.onnx" \
    "wav2vec/wav2vec2-base-960h/config.json"
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
