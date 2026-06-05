#!/bin/bash
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Hallo – RunPod Serverless Worker                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo "  HALLO_WEIGHTS : ${HALLO_WEIGHTS:-/runpod-volume/weights/hallo}"

# Boot up the python listener handler environment 
python3 /app/handler.py
