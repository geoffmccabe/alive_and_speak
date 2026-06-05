"""
RunPod Serverless Handler – FLOAT
https://github.com/deepbrainai-research/float

FLOAT uses flow matching (not diffusion), so generation is extremely fast:
  - Only 10 steps (nfe) by default — completes in seconds to ~1 minute
  - Much cheaper than MultiTalk (~50x faster, ~50x less VRAM)

Input schema:
{
  "input": {
    "image_url":   "https://...",          # REQUIRED — frontal portrait image
    "audio_url":   "https://.../audio.wav",# REQUIRED
    "emotion":     null,                   # optional: angry|disgust|fear|happy|neutral|sad|surprise
                                           # null = infer emotion from audio (S2E mode)
    "no_crop":     false,                  # skip face cropping (use if face is already centered)
    "a_cfg_scale": 2.0,                    # audio cfg scale (default 2.0)
    "e_cfg_scale": 1.0,                    # emotion intensity (try 5-10 for dramatic)
    "r_cfg_scale": 1.0,                    # reference cfg scale
    "nfe":         10,                     # flow steps (default 10, more = slower but smoother)
    "seed":        25
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
from pathlib import Path
from typing import Union

import runpod

# ─────────────────────────────────────────────────────────────────
#  Model paths  (all under /runpod-volume/weights/float/)
# ─────────────────────────────────────────────────────────────────
FLOAT_WEIGHTS      = os.environ.get("FLOAT_WEIGHTS",   "/runpod-volume/weights/float")
CKPT_PATH          = os.environ.get("CKPT_PATH",       f"{FLOAT_WEIGHTS}/float.pth")
OUTPUT_DIR         = os.environ.get("OUTPUT_DIR",      "/tmp/float_outputs")
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "300"))  # 5 min plenty for FLOAT

# Force transformers library to map directly to local layout assets
os.environ["TRANSFORMERS_CACHE"] = FLOAT_WEIGHTS
os.environ["HF_HOME"] = FLOAT_WEIGHTS

# ─────────────────────────────────────────────────────────────────
#  Upload config
# ─────────────────────────────────────────────────────────────────
SIGNED_UPLOAD_ENDPOINT: str = os.environ.get(
    "SIGNED_UPLOAD_ENDPOINT",
    "https://kabdqrzcewkzbjmeqmxx.supabase.co/functions/v1/runpod-signed-upload",
)
RUNPOD_UPLOAD_SECRET: str = os.environ.get(
    "RUNPOD_UPLOAD_SECRET",
    "67mN2pQ9xR4vT8wY3zA5bC6dE1fG0hJ4kL8nM2oP6qS9t",
)

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


def upload_video(video_path: str, job_id: str) -> dict:
    """Upload with 3 retries. Falls back to base64 if all fail."""
    mb       = os.path.getsize(video_path) / 1_000_000
    filename = f"{job_id}.mp4"
    log(f"  ⬆  Uploading {filename} ({mb:.1f} MB)")

    last_error = ""
    for attempt in range(1, 4):
        log(f"     Attempt {attempt}/3 …")
        try:
            with open(video_path, "rb") as fh:
                resp = requests.post(
                    SIGNED_UPLOAD_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {RUNPOD_UPLOAD_SECRET}",
                        "Content-Type":  "video/mp4",
                        "X-Job-Id":      job_id,
                        "X-Filename":    filename,
                    },
                    data=fh,
                    timeout=300,
                )
            log(f"     HTTP {resp.status_code}")
            if resp.ok:
                payload   = resp.json()
                video_url = (payload.get("url") or payload.get("publicUrl")
                             or payload.get("signedUrl") or "")
                log(f"     ✓ {video_url}")
                return {"video_url": video_url, "upload_response": payload,
                        "upload_method": "supabase"}
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.exceptions.Timeout:
            last_error = f"Attempt {attempt} timed out"
        except Exception as e:
            last_error = str(e)
        log(f"     ✗ {last_error}")
        if attempt < 3:
            time.sleep(2 ** attempt)

    # Base64 fallback
    log(f"  ⚠  Upload failed. Encoding as base64 fallback.")
    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "video_url":      "",
        "video_base64":   b64,
        "video_filename": filename,
        "upload_method":  "base64_fallback",
        "upload_error":   last_error,
        "note":           "Decode video_base64 to recover the mp4.",
    }


def find_output_video(save_path: str) -> Union[str, None]:
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return save_path
    # Glob fallback
    stem = Path(save_path).stem
    for p in sorted(Path(OUTPUT_DIR).glob(f"{stem}*.mp4")):
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
    log(f"  Job  : {job_id}")
    log(f"  Keys : {list(inp.keys())}")
    log(f"{'═' * 60}\n")

    with tempfile.TemporaryDirectory(prefix=f"float_{job_id}_") as _tmp:
        tmp = Path(_tmp)

        # ── 1. Download image ─────────────────────────────────────
        image_url = inp.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required", "job_id": job_id}

        img_ext    = Path(image_url.split("?")[0]).suffix or ".jpg"
        image_path = str(tmp / f"ref{img_ext}")
        try:
            download_file(image_url, image_path)
        except Exception as exc:
            return {"error": f"Image download failed: {exc}", "job_id": job_id}

        # ── 2. Download audio ─────────────────────────────────────
        audio_url = inp.get("audio_url") or inp.get("audio_urls")
        if isinstance(audio_url, list):
            audio_url = audio_url[0]
        if not audio_url:
            return {"error": "'audio_url' is required", "job_id": job_id}

        aud_ext    = Path(audio_url.split("?")[0]).suffix or ".wav"
        audio_path = str(tmp / f"audio{aud_ext}")
        try:
            download_file(audio_url, audio_path)
        except Exception as exc:
            return {"error": f"Audio download failed: {exc}", "job_id": job_id}

        # ── 3. Output path ────────────────────────────────────────
        output_path = str(Path(OUTPUT_DIR) / f"{job_id}.mp4")

        # ── 4. Parameters ─────────────────────────────────────────
        emotion      = inp.get("emotion")       # None = infer from audio
        no_crop      = bool(inp.get("no_crop",      False))
        a_cfg_scale  = float(inp.get("a_cfg_scale",  2.0))
        e_cfg_scale  = float(inp.get("e_cfg_scale",  1.0))
        r_cfg_scale  = float(inp.get("r_cfg_scale",  1.0))
        nfe          = int(inp.get("nfe",             10))
        seed         = int(inp.get("seed",           25))

        log(f"  emotion     : {emotion or 'S2E (from audio)'}")
        log(f"  no_crop     : {no_crop}")
        log(f"  nfe         : {nfe}  (flow steps)")
        log(f"  a_cfg_scale : {a_cfg_scale}")
        log(f"  e_cfg_scale : {e_cfg_scale}")

        # Hotfix script code pathways to point straight to local folders instead of HF download queries
        try:
            gen_script = "/app/generate.py"
            if os.path.exists(gen_script):
                with open(gen_script, "r") as f:
                    code = f.read()
                
                changed = False
                if "facebook/wav2vec2-base-960h" in code:
                    code = code.replace("facebook/wav2vec2-base-960h", f"{FLOAT_WEIGHTS}/wav2vec2-base-960h")
                    changed = True
                if "r-f/wav2vec-english-speech-emotion-recognition" in code:
                    code = code.replace("r-f/wav2vec-english-speech-emotion-recognition", f"{FLOAT_WEIGHTS}/wav2vec-english-speech-emotion-recognition")
                    changed = True
                
                if changed:
                    with open(gen_script, "w") as f:
                        f.write(code)
                    log("   ✓ Code patch successfully applied to generate.py paths.")
        except Exception as e:
            log(f"   ⚠ Could not hotfix script text strings: {e}")

        # ── 5. Build CLI command ──────────────────────────────────
        cmd: list[str] = [
            sys.executable,       "/app/generate.py",
            "--ref_path",         image_path,
            "--aud_path",         audio_path,
            "--output_name",      output_path,
            "--ckpt_path",        CKPT_PATH,
            "--a_cfg_scale",      str(a_cfg_scale),
            "--e_cfg_scale",      str(e_cfg_scale),
            "--r_cfg_scale",      str(r_cfg_scale),
            "--nfe",              str(nfe),
            "--seed",             str(seed),
        ]

        if emotion:
            cmd += ["--emo", emotion]
        if no_crop:
            cmd.append("--no_crop")

        log(f"\n  Command:\n  {' '.join(cmd)}\n")

        # ── 6. Run generation ─────────────────────────────────────
        log("  🚀 Launching FLOAT generation…\n")
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
                    return {"error": f"Timed out after {GENERATION_TIMEOUT}s", "job_id": job_id}

            return_code = proc.wait()

        except Exception as exc:
            return {"error": f"Subprocess failed: {exc}", "job_id": job_id}

        log(f"\n  Return code: {return_code}")

        if return_code != 0:
            return {
                "error":      "Video generation failed",
                "returncode": return_code,
                "output":     "\n".join(lines[-60:]),
                "job_id":     job_id,
            }

        # ── 7. Find output video ──────────────────────────────────
        video_path = find_output_video(output_path)
        if not video_path:
            return {
                "error":               "Output video not found",
                "expected_path":       output_path,
                "output_dir_contents": [str(p) for p in Path(OUTPUT_DIR).iterdir()],
                "job_id":              job_id,
            }

        log(f"  ✓ Output: {video_path}")

        # ── 8. Upload ─────────────────────────────────────────────
        upload_result = upload_video(video_path, job_id)

        if upload_result.get("upload_method") == "supabase":
            try:
                os.remove(video_path)
            except Exception:
                pass

        status = "success" if upload_result.get("video_url") else "success_base64_fallback"
        return {"status": status, "job_id": job_id, **upload_result}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
