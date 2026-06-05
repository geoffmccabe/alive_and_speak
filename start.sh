#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  start.sh – Container entrypoint for FLOAT RunPod Serverless
#
#  Models expected at: /runpod-volume/weights/float/
#    float.pth
#    wav2vec2-base-960h/
#    wav2vec-english-speech-emotion-recognition/
#
#  FLOAT reads checkpoints from ./checkpoints/ relative to CWD.
#  We symlink /app/checkpoints → /runpod-volume/weights/float
#  so all internal paths resolve automatically.
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  FLOAT – RunPod Serverless Worker                        ║"
echo "╚══════════════════════════════════════════════════════════╝"

FLOAT_WEIGHTS="${FLOAT_WEIGHTS:-/runpod-volume/weights/float}"

echo ""
echo "  FLOAT_WEIGHTS : ${FLOAT_WEIGHTS}"
echo ""

# ── Symlink /app/checkpoints → volume weights dir ─────────────────
#
#  FLOAT's DataProcessor loads wav2vec from opt.wav2vec_model_path
#  which defaults to ./checkpoints/wav2vec2-base-960h
#  Symlinking makes all relative checkpoint paths resolve correctly.
#
APP_CKPTS="/app/checkpoints"

if [[ -d "${FLOAT_WEIGHTS}" ]]; then
    if [[ -d "${APP_CKPTS}" && ! -L "${APP_CKPTS}" ]]; then
        rm -rf "${APP_CKPTS}"
        echo "  Removed placeholder: ${APP_CKPTS}"
    fi
    if [[ ! -L "${APP_CKPTS}" ]]; then
        ln -sf "${FLOAT_WEIGHTS}" "${APP_CKPTS}"
        echo "  Linked: ${APP_CKPTS} → ${FLOAT_WEIGHTS}"
    else
        echo "  Already linked: ${APP_CKPTS}"
    fi

    # Verify critical files
    echo ""
    echo "  Checking model files:"
    for f in "float.pth" \
              "wav2vec2-base-960h/config.json" \
              "wav2vec-english-speech-emotion-recognition/config.json"; do
        full="${FLOAT_WEIGHTS}/${f}"
        if [[ -f "${full}" ]]; then
            sz=$(du -sh "${full}" | cut -f1)
            echo "    ✓  ${f}  (${sz})"
        else
            echo "    ✗  MISSING: ${f}"
        fi
    done
else
    echo "  ⚠  ${FLOAT_WEIGHTS} not found — is the volume mounted?"
    echo "     Run download_models.sh first."
fi

echo ""
echo "  Starting RunPod handler…"
echo ""

exec python -u /app/handler.py
