"""
RunPod Serverless Handler – Hallo3
https://github.com/fudan-generative-vision/hallo3
CVPR 2025 — CogVideoX-5B DiT backbone

Input format (inference txt file):
  Each line: <prompt>@@<image_path>@@<audio_path>
  Hallo3 parses this in sample_video.py line 260.

Output:
  Saved at: output_dir/<image_stem>-<audio_stem>-seed_<N>/000000_with_audio.mp4

Model layout at /workspace/weights/hallo3/:
  cogvideox-5b-i2v-sat/transformer/1/mp_rank_00_model_states.pt
  cogvideox-5b-i2v-sat/vae/3d-vae.pt
  face_analysis/models/*.onnx + face_landmarker*.task
  hallo3/1/mp_rank_00_model_states.pt + latest
  t5-v1_1-xxl/
  wav2vec/wav2vec2-base-960h/
  audio_separator/Kim_Vocal_2.onnx
"""

import os
import sys
import time
import base64
import requests
import tempfile
import subprocess
import asyncio
from pathlib import Path
from typing import Union

import runpod

HALLO3_WEIGHTS     = os.environ.get("HALLO3_WEIGHTS",  "/workspace/weights/hallo3/pretrained_models")
OUTPUT_DIR         = os.environ.get("OUTPUT_DIR",       "/tmp/hallo3_outputs")
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "9000000"))  # 15 min

os.environ["HF_HOME"]            = HALLO3_WEIGHTS
os.environ["TRANSFORMERS_CACHE"] = HALLO3_WEIGHTS

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def _dl(url: str, dest: str, timeout: int = 180) -> bool:
    try:
        log(f"  ⬇  {url}")
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65_536):
                fh.write(chunk)
        log(f"     ✓ {os.path.getsize(dest)/1e6:.1f} MB")
        return True
    except Exception as e:
        log(f"     ✗ {e}")
        return False


def ensure_models_ready() -> None:
    weights     = Path(HALLO3_WEIGHTS)
    models_dir  = weights / "face_analysis" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # InsightFace onnx files — Hallo3 uses scrfd not buffalo_l
    onnx_files = {
        "scrfd_10g_bnkps.onnx": "https://huggingface.co/deepinsight/insightface/resolve/main/models/scrfd_10g_bnkps.onnx",
        "1k3d68.onnx":          "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/1k3d68.onnx",
        "2d106det.onnx":        "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/2d106det.onnx",
        "genderage.onnx":       "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/genderage.onnx",
        "glintr100.onnx":       "https://huggingface.co/deepinsight/insightface/resolve/main/models/glintr100.onnx",
    }
    for fname, url in onnx_files.items():
        fpath = models_dir / fname
        if not fpath.exists() or fpath.stat().st_size < 1000:
            log(f"  ⚠  Downloading {fname}…")
            _dl(url, str(fpath))
        else:
            log(f"  ✓  {fname}")

    landmarker = models_dir / "face_landmarker_v2_with_blendshapes.task"
    if not landmarker.exists():
        _dl("https://huggingface.co/fudan-generative-ai/hallo2/resolve/main/face_analysis/models/face_landmarker_v2_with_blendshapes.task",
            str(landmarker))

    # wav2vec safetensors → pytorch_model.bin if needed
    wav2vec_dir = weights / "wav2vec" / "wav2vec2-base-960h"
    pt  = wav2vec_dir / "pytorch_model.bin"
    sft = wav2vec_dir / "model.safetensors"
    if sft.exists() and not pt.exists():
        try:
            from safetensors.torch import load_file
            import torch
            torch.save(load_file(str(sft)), str(pt))
            log("  ✓  wav2vec converted")
        except Exception as e:
            log(f"  ⚠  wav2vec conversion failed: {e}")

    # /app/pretrained_models → HALLO3_WEIGHTS
    # Hallo3 configs use ./pretrained_models/... relative paths
    app_pre = Path("/app/pretrained_models")
    if not app_pre.is_symlink():
        if app_pre.exists():
            app_pre.rename("/app/pretrained_models.bak")
        app_pre.symlink_to(weights.resolve())
        log(f"  ✓  Linked /app/pretrained_models → {weights}")
    else:
        log(f"  ✓  /app/pretrained_models linked")

    # Verify critical model files
    critical = [
        "hallo3/1/mp_rank_00_model_states.pt",
        "cogvideox-5b-i2v-sat/vae/3d-vae.pt",
        "t5-v1_1-xxl/config.json",
        "wav2vec/wav2vec2-base-960h/config.json",
        "audio_separator/Kim_Vocal_2.onnx",
    ]
    for f in critical:
        full = weights / f
        if full.exists():
            log(f"  ✓  {f}")
        else:
            log(f"  ✗  MISSING: {f}")

    log("  ✓  Model readiness check complete")


log("⏳ Model readiness check…")
ensure_models_ready()


def make_square_image(src: str, dst: str) -> str:
    """Center-crop to 1:1 or keep 3:2 — Hallo3 handles both internally"""
    try:
        from PIL import Image
        img = Image.open(src).convert("RGB")
        w, h = img.size
        log(f"  📐 Image: {w}×{h}")
        if w != h:
            s = min(w, h)
            img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
        img.save(dst, "JPEG", quality=95)
        log(f"  ✓  Prepared square image")
        return dst
    except Exception as e:
        log(f"  ⚠  Crop failed: {e}")
        return src


def find_output_video(output_dir: str) -> Union[str, None]:
    """Hallo3 saves: output_dir/<name>/000000_with_audio.mp4"""
    base = Path(output_dir)
    # First check subdirectories for *_with_audio.mp4
    for p in sorted(base.rglob("*_with_audio.mp4")):
        if p.stat().st_size > 0:
            return str(p)
    # Fallback: any mp4
    for p in sorted(base.rglob("*.mp4")):
        if p.stat().st_size > 0:
            return str(p)
    return None


def denoise_video(input_path: str, output_path: str) -> str:
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vf", "hqdn3d=4:3:6:4.5",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy", output_path,
        ], capture_output=True, text=True)
        if result.returncode == 0 and Path(output_path).stat().st_size > 0:
            log(f"  ✓  Denoised")
            return output_path
        return input_path
    except Exception as e:
        log(f"  ⚠  Denoise failed: {e}")
        return input_path


def handler(job: dict) -> dict:
    job_id = job["id"]
    inp    = job["input"]

    log(f"\n{'═'*60}")
    log(f"  Hallo3 Job : {job_id}")
    log(f"  Keys       : {list(inp.keys())}")
    log(f"{'═'*60}\n")

    with tempfile.TemporaryDirectory(prefix=f"hallo3_{job_id}_") as _tmp:
        tmp = Path(_tmp)
        job_output_dir = tmp / "output"
        job_output_dir.mkdir()

        # ── Download image ────────────────────────────────────────
        image_url = inp.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required", "job_id": job_id}
        img_ext   = Path(image_url.split("?")[0]).suffix or ".jpg"
        raw_img   = str(tmp / f"raw{img_ext}")
        sq_img    = str(tmp / "portrait.jpg")
        try:
            r = requests.get(image_url, stream=True, timeout=120)
            r.raise_for_status()
            with open(raw_img, "wb") as f:
                for chunk in r.iter_content(65536): f.write(chunk)
            log(f"     ✓ {os.path.getsize(raw_img)/1e6:.1f} MB")
        except Exception as e:
            return {"error": f"Image download failed: {e}", "job_id": job_id}
        image_path = make_square_image(raw_img, sq_img)

        # ── Download audio ────────────────────────────────────────
        audio_url = inp.get("audio_url")
        if not audio_url:
            return {"error": "'audio_url' is required", "job_id": job_id}
        aud_ext    = Path(audio_url.split("?")[0]).suffix or ".wav"
        audio_path = str(tmp / f"audio{aud_ext}")
        try:
            r = requests.get(audio_url, stream=True, timeout=120)
            r.raise_for_status()
            with open(audio_path, "wb") as f:
                for chunk in r.iter_content(65536): f.write(chunk)
            log(f"     ✓ {os.path.getsize(audio_path)/1e6:.1f} MB")
        except Exception as e:
            return {"error": f"Audio download failed: {e}", "job_id": job_id}

        # Optional prompt — Hallo3 supports text conditioning
        prompt = inp.get("prompt", "A person talking.")

        # ── Write input.txt ───────────────────────────────────────
        # Format: <prompt>@@<image_path>@@<audio_path>
        input_txt = str(tmp / "input.txt")
        with open(input_txt, "w") as f:
            f.write(f"{prompt}@@{image_path}@@{audio_path}\n")
        log(f"  Input: {prompt}@@{image_path}@@{audio_path}")

        # ── Run inference ─────────────────────────────────────────
        cmd = [
            "bash", "/app/scripts/inference_long_batch.sh",
            input_txt,
            str(job_output_dir),
        ]
        log(f"  Command: {' '.join(cmd)}\n")
        log("  🚀 Launching Hallo3 DiT pipeline…\n")

        try:
            proc = subprocess.Popen(
                cmd, cwd="/app",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            start = time.time()
            lines = []
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                log(line)
                lines.append(line)
                if time.time() - start > GENERATION_TIMEOUT:
                    proc.terminate()
                    proc.wait(timeout=10)
                    return {"error": f"Timed out after {GENERATION_TIMEOUT}s", "job_id": job_id}
            return_code = proc.wait()
        except Exception as e:
            return {"error": f"Subprocess failed: {e}", "job_id": job_id}

        log(f"\n  Return code: {return_code}")

        video_path = find_output_video(str(job_output_dir))

        if return_code != 0 and not video_path:
            return {
                "error":      "Inference failed",
                "returncode": return_code,
                "output":     "\n".join(lines[-60:]),
                "job_id":     job_id,
            }

        if not video_path:
            return {
                "error":  "Output video not found",
                "job_id": job_id,
                "dir":    [str(p) for p in job_output_dir.rglob("*")],
            }

        log(f"  ✓ Output: {video_path}")

        # Denoise
        denoised = str(tmp / f"{job_id}_denoised.mp4")
        video_path = denoise_video(video_path, denoised)

        mb = os.path.getsize(video_path) / 1e6
        log(f"  ⚙  Encoding {mb:.1f} MB…")

        try:
            with open(video_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return {
                "status":         "success",
                "job_id":         job_id,
                "video_base64":   b64,
                "video_filename": f"{job_id}.mp4",
            }
        except Exception as e:
            return {"error": f"Encoding failed: {e}", "job_id": job_id}


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    runpod.serverless.start({"handler": handler})
