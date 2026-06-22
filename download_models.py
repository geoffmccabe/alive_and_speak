"""
Downloads all FLOAT model files into /app/checkpoints/ during Docker build.
Run as: python download_models.py
"""
import os
import sys
from huggingface_hub import hf_hub_download

CKPTS = "/app/checkpoints"
os.makedirs(CKPTS, exist_ok=True)

# ── 1. wav2vec2-base-960h (audio encoder) ────────────────────────
print("\n── Downloading facebook/wav2vec2-base-960h ──")
wav2vec_files = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
]
dest = os.path.join(CKPTS, "wav2vec2-base-960h")
os.makedirs(dest, exist_ok=True)
for f in wav2vec_files:
    try:
        hf_hub_download(
            repo_id="facebook/wav2vec2-base-960h",
            filename=f,
            local_dir=dest,
        )
        mb = os.path.getsize(os.path.join(dest, f)) / 1_000_000
        print(f"  ✓  {f}  ({mb:.1f} MB)")
    except Exception as e:
        print(f"  ✗  {f}: {e}")
        sys.exit(1)

# ── 2. wav2vec-english-speech-emotion-recognition ────────────────
print("\n── Downloading r-f/wav2vec-english-speech-emotion-recognition ──")
emotion_files = [
    "config.json",
    "pytorch_model.bin",
    "preprocessor_config.json",
]
dest = os.path.join(CKPTS, "wav2vec-english-speech-emotion-recognition")
os.makedirs(dest, exist_ok=True)
for f in emotion_files:
    try:
        hf_hub_download(
            repo_id="r-f/wav2vec-english-speech-emotion-recognition",
            filename=f,
            local_dir=dest,
        )
        mb = os.path.getsize(os.path.join(dest, f)) / 1_000_000
        print(f"  ✓  {f}  ({mb:.1f} MB)")
    except Exception as e:
        print(f"  ✗  {f}: {e}")
        sys.exit(1)

# ── 3. Verify all required files ─────────────────────────────────
print("\n── Verifying all model files ──")
required = [
    "float.pth",
    "wav2vec2-base-960h/config.json",
    "wav2vec2-base-960h/model.safetensors",
    "wav2vec2-base-960h/preprocessor_config.json",
    "wav2vec2-base-960h/tokenizer_config.json",
    "wav2vec2-base-960h/vocab.json",
    "wav2vec2-base-960h/special_tokens_map.json",
    "wav2vec-english-speech-emotion-recognition/config.json",
    "wav2vec-english-speech-emotion-recognition/pytorch_model.bin",
    "wav2vec-english-speech-emotion-recognition/preprocessor_config.json",
]
missing = []
for f in required:
    path = os.path.join(CKPTS, f)
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1_000_000
        print(f"  ✓  {f}  ({mb:.1f} MB)")
    else:
        print(f"  ✗  MISSING: {f}")
        missing.append(f)

if missing:
    print(f"\nBUILD FAILED: {len(missing)} file(s) missing: {missing}")
    sys.exit(1)

print("\nAll model files present — download complete.")
