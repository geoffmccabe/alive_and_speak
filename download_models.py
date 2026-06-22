"""
Downloads all FLOAT model files into /app/checkpoints/ during Docker build.
"""
import os
import sys
from huggingface_hub import hf_hub_download

CKPTS = "/app/checkpoints"
os.makedirs(CKPTS, exist_ok=True)

errors = []

def download(repo, filename, dest_dir):
    dest = os.path.join(CKPTS, dest_dir)
    os.makedirs(dest, exist_ok=True)
    try:
        hf_hub_download(repo_id=repo, filename=filename, local_dir=dest)
        mb = os.path.getsize(os.path.join(dest, filename)) / 1_000_000
        print(f"  ✓  {dest_dir}/{filename}  ({mb:.1f} MB)")
    except Exception as e:
        print(f"  ✗  {dest_dir}/{filename}: {e}")
        errors.append(f"{dest_dir}/{filename}")

# ── 1. wav2vec2-base-960h ─────────────────────────────────────────
print("\n── facebook/wav2vec2-base-960h ──")
for f in [
    "config.json",
    "pytorch_model.bin",          # correct filename (NOT model.safetensors)
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
]:
    download("facebook/wav2vec2-base-960h", f, "wav2vec2-base-960h")

# ── 2. wav2vec emotion recognition ───────────────────────────────
print("\n── r-f/wav2vec-english-speech-emotion-recognition ──")
for f in [
    "config.json",
    "pytorch_model.bin",
    "preprocessor_config.json",
]:
    download("r-f/wav2vec-english-speech-emotion-recognition", f,
             "wav2vec-english-speech-emotion-recognition")

# ── Verify ────────────────────────────────────────────────────────
print("\n── Verification ──")
required = [
    "wav2vec2-base-960h/config.json",
    "wav2vec2-base-960h/pytorch_model.bin",
    "wav2vec2-base-960h/preprocessor_config.json",
    "wav2vec2-base-960h/tokenizer_config.json",
    "wav2vec2-base-960h/vocab.json",
    "wav2vec2-base-960h/special_tokens_map.json",
    "wav2vec-english-speech-emotion-recognition/config.json",
    "wav2vec-english-speech-emotion-recognition/pytorch_model.bin",
    "wav2vec-english-speech-emotion-recognition/preprocessor_config.json",
]
for f in required:
    path = os.path.join(CKPTS, f)
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1_000_000
        print(f"  ✓  {f}  ({mb:.1f} MB)")
    else:
        print(f"  ✗  MISSING: {f}")
        errors.append(f)

if errors:
    print(f"\nBUILD FAILED — {len(errors)} missing: {errors}")
    sys.exit(1)

print("\nAll HuggingFace models downloaded successfully.")
