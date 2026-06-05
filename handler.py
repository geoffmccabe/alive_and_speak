"""
RunPod Serverless Handler – Hallo
Hierarchical Audio-Driven Visual Synthesis
Input schema:
{
  "input": {
    "image_url":   "https://...",          # REQUIRED — Source portrait image
    "audio_url":   "https://.../audio.wav",# REQUIRED — Voice audio file
    "pose_weight": 1.0,                    # optional: scale weight for motion/pose
    "face_weight": 1.0,                    # optional: scale weight for facial expressions
    "lip_weight":  1.0,                    # optional: scale weight for lips matching voice
    "steps":       40                      # optional: diffusion steps (higher = smoother)
  }
}
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

# ─────────────────────────────────────────────────────────────────
#  Model paths  (all under /runpod-volume/weights/hallo/)
# ─────────────────────────────────────────────────────────────────
HALLO_WEIGHTS      = os.environ.get("HALLO_WEIGHTS",   "/runpod-volume/weights/hallo")
OUTPUT_DIR         = os.environ.get("OUTPUT_DIR",      "/tmp/hallo_outputs")
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "450")) # 7.5 mins matching diffusion needs

# Direct Cache systems locally 
os.environ["TRANSFORMERS_CACHE"] = HALLO_WEIGHTS
os.environ["HF_HOME"] = HALLO_WEIGHTS
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(msg, flush=True)

def download_file(url: str, dest: str, timeout: int = 120) -> str:
    log(f"  ⬇  {url}")
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=65_536):
            fh.write(chunk)
    mb = os.path.getsize(dest) / 1_000_000
    log(f"     ✓ {mb:.1f} MB")
    return dest

def find_output_video(save_dir: str, prefix: str) -> Union[str, None]:
    for p in sorted(Path(save_dir).glob(f"*{prefix}*.mp4")):
        if p.stat().st_size > 0:
            return str(p)
    return None

# ─────────────────────────────────────────────────────────────────
#  Main handler
# ─────────────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    job_id = job["id"]
    inp    = job["input"]

    log(f"\n{'═' * 60}")
    log(f"  Hallo Job : {job_id}")
    log(f"  Keys      : {list(inp.keys())}")
    log(f"{'═' * 60}\n")

    with tempfile.TemporaryDirectory(prefix=f"hallo_{job_id}_") as _tmp:
        tmp = Path(_tmp)

        # ── 1. Download reference image ───────────────────────────
        image_url = inp.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required", "job_id": job_id}

        img_ext    = Path(image_url.split("?")[0]).suffix or ".jpg"
        image_path = str(tmp / f"source_portrait{img_ext}")
        try:
            download_file(image_url, image_path)
        except Exception as exc:
            return {"error": f"Image download failed: {exc}", "job_id": job_id}

        # ── 2. Download audio vocal track ─────────────────────────
        audio_url = inp.get("audio_url")
        if not audio_url:
            return {"error": "'audio_url' is required", "job_id": job_id}

        aud_ext    = Path(audio_url.split("?")[0]).suffix or ".wav"
        audio_path = str(tmp / f"driving_vocal{aud_ext}")
        try:
            download_file(audio_url, audio_path)
        except Exception as exc:
            return {"error": f"Audio download failed: {exc}", "job_id": job_id}

        # ── 3. Handle Hyperparameters ─────────────────────────────
        pose_weight = float(inp.get("pose_weight", 1.0))
        face_weight = float(inp.get("face_weight", 1.0))
        lip_weight  = float(inp.get("lip_weight",  1.0))
        steps       = int(inp.get("steps",         40))

        # ── 4. Build CLI inference command for Hallo repo ─────────
        cmd: list[str] = [
            sys.executable,     "scripts/inference.py",
            "--source_image",   image_path,
            "--driving_audio",  audio_path,
            "--output_path",    str(Path(OUTPUT_DIR) / f"{job_id}.mp4"),
            "--pose_weight",    str(pose_weight),
            "--face_weight",    str(face_weight),
            "--lip_weight",     str(lip_weight),
            "--inference_steps", str(steps),
            "--ckpt_path",      f"{HALLO_WEIGHTS}/hallo/net.pth"
        ]

        log(f"\n  Command:\n  {' '.join(cmd)}\n")

        # ── 5. Execute Hallo Inference ────────────────────────────
        log("  🚀 Launching Hallo Diffusion Pipeline Generation…\n")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd="/app",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            start = time.time()
            lines: list[str] = []
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                log(line)
                lines.append(line)
                if time.time() - start > GENERATION_TIMEOUT:
                    proc.terminate()
                    proc.wait(timeout=10)
                    return {"error": f"Inference execution timed out after {GENERATION_TIMEOUT}s", "job_id": job_id}

            return_code = proc.wait()

        except Exception as exc:
            return {"error": f"Subprocess running scripts/inference.py failed: {exc}", "job_id": job_id}

        log(f"\n  Return code: {return_code}")

        if return_code != 0:
            return {
                "error":      "Video animation synthesis failed",
                "returncode": return_code,
                "output":     "\n".join(lines[-40:]),
                "job_id":     job_id,
            }

        # ── 6. Check output file ──────────────────────────────────
        expected_path = str(Path(OUTPUT_DIR) / f"{job_id}.mp4")
        video_path = find_output_video(OUTPUT_DIR, job_id) or expected_path
        
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return {
                "error":               "Output animated video missing or empty",
                "output_dir_contents": [str(p) for p in Path(OUTPUT_DIR).iterdir()],
                "job_id":              job_id,
            }

        # ── 7. Direct Base64 Encoding ─────────────────────────────
        mb = os.path.getsize(video_path) / 1_000_000
        log(f"  ⚙  Encoding generated video to Base64 output string ({mb:.1f} MB)…")
        
        try:
            with open(video_path, "rb") as f:
                b64_string = base64.b64encode(f.read()).decode("utf-8")
            
            # Delete local file to optimize local ephemeral storage space
            try:
                os.remove(video_path)
            except Exception:
                pass
                
            return {
                "status": "success",
                "job_id": job_id,
                "video_base64": b64_string,
                "video_filename": f"{job_id}.mp4"
            }
        except Exception as err:
            return {
                "error": f"Failed to encode video output buffer to Base64 sequence string: {err}",
                "job_id": job_id
            }

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    runpod.serverless.start({"handler": handler})
