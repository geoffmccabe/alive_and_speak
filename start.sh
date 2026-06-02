#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  start.sh – Container entrypoint for MultiTalk RunPod Serverless
#
#  Responsibilities:
#    1. Symlink MultiTalk adapter weights into the Wan2.1 checkpoint
#       directory (models live on the network volume, not in image).
#    2. Launch the RunPod serverless handler.
#
#  Override any path via environment variable before starting the
#  container, e.g.:
#    -e CKPT_DIR=/runpod-volume/models/Wan2.1-I2V-14B-480P
#    -e MULTITALK_DIR=/runpod-volume/models/MeiGen-MultiTalk
#    -e WAV2VEC_DIR=/runpod-volume/models/chinese-wav2vec2-base
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MultiTalk – RunPod Serverless Worker                    ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Model path defaults (match RunPod network volume layout) ──────
CKPT_DIR="${CKPT_DIR:-/runpod-volume/models/Wan2.1-I2V-14B-480P}"
MULTITALK_DIR="${MULTITALK_DIR:-/runpod-volume/models/MeiGen-MultiTalk}"
WAV2VEC_DIR="${WAV2VEC_DIR:-/runpod-volume/models/chinese-wav2vec2-base}"

echo ""
echo "  CKPT_DIR      : ${CKPT_DIR}"
echo "  MULTITALK_DIR : ${MULTITALK_DIR}"
echo "  WAV2VEC_DIR   : ${WAV2VEC_DIR}"
echo ""

# ── Symlink MultiTalk adapter weights into Wan2.1 dir ─────────────
# MultiTalk ships two extra files that must appear inside the Wan2.1
# checkpoint folder:
#   • multitalk.safetensors             (adapter weights)
#   • diffusion_pytorch_model.safetensors.index.json  (updated index)
#
# We symlink from the volume instead of copying to save disk space.

if [[ -d "${CKPT_DIR}" && -d "${MULTITALK_DIR}" ]]; then
    echo "  Setting up MultiTalk weight symlinks…"

    ORIG_IDX="${CKPT_DIR}/diffusion_pytorch_model.safetensors.index.json"
    BKUP_IDX="${ORIG_IDX}_orig"
    MT_IDX="${MULTITALK_DIR}/diffusion_pytorch_model.safetensors.index.json"

    # Back up the original Wan2.1 index once (idempotent)
    if [[ -f "${ORIG_IDX}" && ! -f "${BKUP_IDX}" && ! -L "${ORIG_IDX}" ]]; then
        mv "${ORIG_IDX}" "${BKUP_IDX}"
        echo "    Backed up original index → $(basename ${BKUP_IDX})"
    fi

    # Symlink MultiTalk index
    if [[ -f "${MT_IDX}" && ! -L "${ORIG_IDX}" ]]; then
        ln -sf "${MT_IDX}" "${ORIG_IDX}"
        echo "    Linked: diffusion_pytorch_model.safetensors.index.json"
    fi

    # Symlink MultiTalk adapter weights
    MT_W="${MULTITALK_DIR}/multitalk.safetensors"
    DEST_W="${CKPT_DIR}/multitalk.safetensors"
    if [[ -f "${MT_W}" && ! -L "${DEST_W}" ]]; then
        ln -sf "${MT_W}" "${DEST_W}"
        echo "    Linked: multitalk.safetensors"
    fi

    echo "    ✓ Symlinks ready"
else
    echo "  ⚠  WARNING: One or both model directories not found."
    echo "     CKPT_DIR exists      : $([ -d "${CKPT_DIR}" ]      && echo YES || echo NO)"
    echo "     MULTITALK_DIR exists : $([ -d "${MULTITALK_DIR}" ] && echo YES || echo NO)"
    echo "     Handler will start anyway — jobs will fail until volumes are mounted."
fi

echo ""
echo "  Starting RunPod handler…"
echo ""

# exec replaces the shell process so signals are forwarded cleanly
exec python -u /app/handler.py
