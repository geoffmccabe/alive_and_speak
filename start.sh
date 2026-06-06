#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  start.sh – Hallo RunP
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Hallo – RunPod Serverless Worker                        ║"
echo "╚══════════════════════════════════════════════════════════╝"

HALLO_WEIGHTS="${HALLO_WEIGHTS:-/runpod-volume/weights/hallo}"
echo ""
echo "  HALLO_WEIGHTS : ${HALLO_WEIGHTS}"
echo ""

# ── Verify critical model files are present ───────────────────────
echo "  Checking model files:"
MISSING=0
for f in \
    "hallo/net.pth" \
    "motion_module/mm_sd_v15_v2.ckpt" \
    "sd-vae-ft-mse/config.json" \
    "stable-diffusion-v1-5/unet/config.json" \
    "face_analysis/models/buffalo_l/det_10g.onnx" \
    "wav2vec/wav2vec2-base-960h/config.json"
do
    full="${HALLO_WEIGHTS}/${f}"
    if [[ -f "${full}" ]]; then
        sz=$(du -sh "${full}" 2>/dev/null | cut -f1)
        echo "    ✓  ${f}  (${sz})"
    else
        echo "    ✗  MISSING: ${full}"
        MISSING=1
    fi
done

if [[ "${MISSING}" == "1" ]]; then
    echo ""
    echo "  ⚠  One or more model files are missing."
    echo "     Run the download command shown below, then restart."
    echo ""
fi

echo ""
echo "  Starting RunPod serverless handler…"
echo ""

exec python3 -u /app/handler.py
