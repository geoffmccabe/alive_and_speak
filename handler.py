"""
RunPod Serverless Handler – Hallo
https://github.com/fudan-generative-vision/hallo

Config structure (from configs/inference/default.yaml):
  face_analysis.model_path      ← nested, NOT face_analysis_model_path
  vae.model_path
  wav2vec.model_path
  audio_separator.model_path
  audio_ckpt_dir                ← flat
  base_model_path               ← flat
  motion_module_path            ← flat
"""

import os
import sys
import time
import base64
import requests
import tempfile
import subprocess
import asyncio
import yaml
from pathlib import Path
from typing import Union

import runpod

HALLO_WEIGHTS      = os.environ.get("HALLO_WEIGHTS", "/runpod-volume/weights/hallo")
OUTPUT_DIR         = os.environ.get("OUTPUT_DIR",    "/tmp/hallo_outputs")
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "600"))

os.environ["HF_HOME"]            = HALLO_WEIGHTS
os.environ["TRANSFORMERS_CACHE"] = HALLO_WEIGHTS

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def fix_insightface_cwd_path() -> None:
    """
    inference.py runs with cwd=/app. InsightFace resolves onnx paths relative
    to cwd when face_analysis.model_path is a relative path like ./pretrained_models/...
    We symlink /app/pretrained_models → volume so all relative paths resolve correctly.
    """
    target = Path("/app/pretrained_models")
    source = Path(HALLO_WEIGHTS)

    if target.is_symlink():
        log(f"  ✓ /app/pretrained_models already linked")
        return

    if target.exists():
        target.rename("/app/pretrained_models.bak")

    target.symlink_to(source)
    log(f"  ✓ Linked /app/pretrained_models → {source}")


fix_insightface_cwd_path()


def download_file(url: str, dest: str, timeout: int = 120) -> str:
    log(f"  ⬇  {url}")
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=65_536):
            fh.write(chunk)
    log(f"     ✓ {os.path.getsize(dest)/1e6:.1f} MB")
    return dest


def find_output_video(save_dir: str, job_id: str) -> Union[str, None]:
    exact = Path(save_dir) / f"{job_id}.mp4"
    if exact.exists() and exact.stat().st_size > 0:
        return str(exact)
    for p in sorted(Path(save_dir).glob("*.mp4")):
        if p.stat().st_size > 0 and job_id[:8] in p.name:
            return str(p)
    return None


def write_job_config(image_path, audio_path, output_path,
                     pose_weight, face_weight, lip_weight,
                     face_expand_ratio, steps, tmp_dir) -> str:
    base_cfg = "/app/configs/inference/default.yaml"
    if not os.path.exists(base_cfg):
        raise FileNotFoundError(f"Missing: {base_cfg}")

    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)

    # ── Top-level inputs/outputs ──────────────────────────────────
    cfg["source_image"] = image_path
    cfg["driving_audio"] = audio_path
    cfg["save_path"]    = output_path

    # ── Flat weight paths ─────────────────────────────────────────
    cfg["base_model_path"]    = f"{HALLO_WEIGHTS}/stable-diffusion-v1-5"
    cfg["motion_module_path"] = f"{HALLO_WEIGHTS}/motion_module/mm_sd_v15_v2.ckpt"
    cfg["audio_ckpt_dir"]     = f"{HALLO_WEIGHTS}/hallo"

    # ── Nested weight paths (must match yaml structure exactly) ───
    if not isinstance(cfg.get("face_analysis"), dict):
        cfg["face_analysis"] = {}
    cfg["face_analysis"]["model_path"] = f"{HALLO_WEIGHTS}/face_analysis"

    if not isinstance(cfg.get("wav2vec"), dict):
        cfg["wav2vec"] = {}
    cfg["wav2vec"]["model_path"] = f"{HALLO_WEIGHTS}/wav2vec/wav2vec2-base-960h"
    cfg["wav2vec"].setdefault("features", "all")

    if not isinstance(cfg.get("vae"), dict):
        cfg["vae"] = {}
    cfg["vae"]["model_path"] = f"{HALLO_WEIGHTS}/sd-vae-ft-mse"

    if not isinstance(cfg.get("audio_separator"), dict):
        cfg["audio_separator"] = {}
    cfg["audio_separator"]["model_path"] = f"{HALLO_WEIGHTS}/audio_separator/Kim_Vocal_2.onnx"

    # ── Hyperparams ───────────────────────────────────────────────
    cfg["inference_steps"]   = steps
    cfg["pose_weight"]       = pose_weight
    cfg["face_weight"]       = face_weight
    cfg["lip_weight"]        = lip_weight
    cfg["face_expand_ratio"] = face_expand_ratio

    cfg_path = str(tmp_dir / "job_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    # Log the resolved paths so we can verify in logs
    log(f"  face_analysis.model_path : {cfg['face_analysis']['model_path']}")
    log(f"  vae.model_path           : {cfg['vae']['model_path']}")
    log(f"  audio_ckpt_dir           : {cfg['audio_ckpt_dir']}")

    return cfg_path


def handler(job: dict) -> dict:
    job_id = job["id"]
    inp    = job["input"]

    log(f"\n{'═'*60}")
    log(f"  Hallo Job : {job_id}")
    log(f"  Keys      : {list(inp.keys())}")
    log(f"{'═'*60}\n")

    with tempfile.TemporaryDirectory(prefix=f"hallo_{job_id}_") as _tmp:
        tmp = Path(_tmp)

        image_url = inp.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required", "job_id": job_id}
        img_ext    = Path(image_url.split("?")[0]).suffix or ".jpg"
        image_path = str(tmp / f"source_portrait{img_ext}")
        try:
            download_file(image_url, image_path)
        except Exception as e:
            return {"error": f"Image download failed: {e}", "job_id": job_id}

        audio_url = inp.get("audio_url")
        if not audio_url:
            return {"error": "'audio_url' is required", "job_id": job_id}
        aud_ext    = Path(audio_url.split("?")[0]).suffix or ".wav"
        audio_path = str(tmp / f"driving_audio{aud_ext}")
        try:
            download_file(audio_url, audio_path)
        except Exception as e:
            return {"error": f"Audio download failed: {e}", "job_id": job_id}

        pose_weight       = float(inp.get("pose_weight",       1.0))
        face_weight       = float(inp.get("face_weight",       1.0))
        lip_weight        = float(inp.get("lip_weight",        1.0))
        face_expand_ratio = float(inp.get("face_expand_ratio", 1.2))
        steps             = int(inp.get("steps",               40))

        log(f"  pose={pose_weight}  face={face_weight}  lip={lip_weight}  "
            f"expand={face_expand_ratio}  steps={steps}")

        output_path = str(Path(OUTPUT_DIR) / f"{job_id}.mp4")

        try:
            cfg_path = write_job_config(
                image_path, audio_path, output_path,
                pose_weight, face_weight, lip_weight,
                face_expand_ratio, steps, tmp)
        except Exception as e:
            return {"error": f"Config failed: {e}", "job_id": job_id}

        cmd = [sys.executable, "scripts/inference.py", "--config", cfg_path]
        log(f"\n  Command: {' '.join(cmd)}\n")
        log("  🚀 Launching Hallo…\n")

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

        if return_code != 0:
            return {
                "error":      "Inference failed",
                "returncode": return_code,
                "output":     "\n".join(lines[-60:]),
                "job_id":     job_id,
            }

        video_path = find_output_video(OUTPUT_DIR, job_id)
        if not video_path:
            return {
                "error":  "Output video not found",
                "job_id": job_id,
                "dir":    [str(p) for p in Path(OUTPUT_DIR).iterdir()],
            }

        log(f"  ✓ {video_path}")
        mb = os.path.getsize(video_path) / 1e6
        log(f"  ⚙  Encoding {mb:.1f} MB…")

        try:
            with open(video_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            try:
                os.remove(video_path)
            except Exception:
                pass
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
