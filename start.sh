#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  start.sh – Container entrypoint for MultiTalk RunPod Serverless
#
#  Models expected at: /workspace/weights/
#    /workspace/weights/Wan2.1-I2V-14B-480P/
#    /workspace/weights/chinese-wav2vec2-base/
#    /workspace/weights/Kokoro-82M/
#    /workspace/weights/MeiGen-MultiTalk/
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MultiTalk – RunPod Serverless Worker                    ║"
echo "╚══════════════════════════════════════════════════════════╝"

WEIGHTS_ROOT="${WEIGHTS_ROOT:-/workspace/weights}"
CKPT_DIR="${CKPT_DIR:-${WEIGHTS_ROOT}/Wan2.1-I2V-14B-480P}"
MULTITALK_DIR="${MULTITALK_DIR:-${WEIGHTS_ROOT}/MeiGen-MultiTalk}"

echo ""
echo "  WEIGHTS_ROOT  : ${WEIGHTS_ROOT}"
echo "  CKPT_DIR      : ${CKPT_DIR}"
echo "  MULTITALK_DIR : ${MULTITALK_DIR}"
echo ""

# ── STEP 1: Symlink /app/weights → /workspace/weights ────────────
#
#  generate_multitalk.py uses hardcoded relative paths like
#  weights/Kokoro-82M and weights/chinese-wav2vec2-base internally.
#  Symlinking /app/weights to our volume weights dir makes all of
#  those resolve correctly without changing the script.
#
APP_WEIGHTS="/app/weights"
if [[ -d "${WEIGHTS_ROOT}" ]]; then
    if [[ -d "${APP_WEIGHTS}" && ! -L "${APP_WEIGHTS}" ]]; then
        rm -rf "${APP_WEIGHTS}"
        echo "  Removed placeholder: ${APP_WEIGHTS}"
    fi
    if [[ ! -L "${APP_WEIGHTS}" ]]; then
        ln -sf "${WEIGHTS_ROOT}" "${APP_WEIGHTS}"
        echo "  Linked: ${APP_WEIGHTS} → ${WEIGHTS_ROOT}"
    else
        echo "  Already linked: ${APP_WEIGHTS}"
    fi
else
    echo "  ⚠  ${WEIGHTS_ROOT} not found — is the volume mounted?"
fi

# ── STEP 2: Symlink MultiTalk adapter files into Wan2.1 dir ──────
#
#  The model loader expects these two files inside the Wan2.1 dir:
#    • multitalk.safetensors
#    • diffusion_pytorch_model.safetensors.index.json  (replaced)
#
if [[ -d "${CKPT_DIR}" && -d "${MULTITALK_DIR}" ]]; then
    echo "  Setting up MultiTalk adapter symlinks…"

    ORIG_IDX="${CKPT_DIR}/diffusion_pytorch_model.safetensors.index.json"
    BKUP_IDX="${ORIG_IDX}_orig"
    MT_IDX="${MULTITALK_DIR}/diffusion_pytorch_model.safetensors.index.json"
    MT_W="${MULTITALK_DIR}/multitalk.safetensors"
    DEST_W="${CKPT_DIR}/multitalk.safetensors"

    # Backup original Wan2.1 index once
    if [[ -f "${ORIG_IDX}" && ! -f "${BKUP_IDX}" && ! -L "${ORIG_IDX}" ]]; then
        mv "${ORIG_IDX}" "${BKUP_IDX}"
        echo "    Backed up original index.json"
    fi

    # Symlink MultiTalk index (must use absolute path)
    if [[ -f "${MT_IDX}" && ! -L "${ORIG_IDX}" ]]; then
        ln -sf "${MT_IDX}" "${ORIG_IDX}"
        echo "    Linked: diffusion_pytorch_model.safetensors.index.json"
    elif [[ -L "${ORIG_IDX}" ]]; then
        echo "    Already linked: index.json"
    else
        echo "    ⚠  ${MT_IDX} not found"
    fi

    # Symlink MultiTalk adapter weights (must use absolute path)
    if [[ -f "${MT_W}" && ! -L "${DEST_W}" ]]; then
        ln -sf "${MT_W}" "${DEST_W}"
        echo "    Linked: multitalk.safetensors"
    elif [[ -L "${DEST_W}" ]]; then
        echo "    Already linked: multitalk.safetensors"
    else
        echo "    ⚠  ${MT_W} not found"
    fi

    echo "    ✓ Adapter setup complete"
else
    echo "  ⚠  Model dirs not ready:"
    echo "     CKPT_DIR      : $([ -d "${CKPT_DIR}" ]      && echo EXISTS || echo MISSING)"
    echo "     MULTITALK_DIR : $([ -d "${MULTITALK_DIR}" ] && echo EXISTS || echo MISSING)"
fi

echo ""
echo "  Starting RunPod handler…"
echo ""

exec python -u /app/handler.py
