"""
RunPod Serverless Handler – Hallo
https://github.com/fudan-generative-vision/hallo

Critical concurrent job fix:
  inference.py line 345: output_file = config.output
  config.output defaults to: save_path + "/output.mp4"
  If save_path is shared (/app/.cache) across concurrent jobs,
  they overwrite each other → noise/artifacts in output.
  Fix: each job gets its own unique save_path directory.
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


def _download(url: str, dest: str, timeout: int = 180) -> bool:
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
        log(f"     ✗ Failed: {e}")
        return False


def ensure_models_ready() -> None:
    weights     = Path(HALLO_WEIGHTS)
    models_dir  = weights / "face_analysis" / "models"
    buffalo_dir = models_dir / "buffalo_l"

    models_dir.mkdir(parents=True, exist_ok=True)
    buffalo_dir.mkdir(parents=True, exist_ok=True)

    # Buffalo_l onnx files
    onnx_files = {
        "det_10g.onnx":   "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
        "1k3d68.onnx":    "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/1k3d68.onnx",
        "2d106det.onnx":  "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/2d106det.onnx",
        "genderage.onnx": "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/genderage.onnx",
        "w600k_r50.onnx": "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
    }
    for fname, url in onnx_files.items():
        fpath = buffalo_dir / fname
        if not fpath.exists() or fpath.stat().st_size < 1000:
            log(f"  ⚠  Missing {fname} — downloading…")
            _download(url, str(fpath))

    # Flat symlinks into models/
    for src in buffalo_dir.glob("*.onnx"):
        dst = models_dir / src.name
        if not dst.exists() and not dst.is_symlink():
            dst.symlink_to(src.resolve())

    # face_landmarker
    landmarker = models_dir / "face_landmarker_v2_with_blendshapes.task"
    if not landmarker.exists():
        _download(
            "https://huggingface.co/fudan-generative-ai/hallo2/resolve/main/face_analysis/models/face_landmarker_v2_with_blendshapes.task",
            str(landmarker)
        )

    # wav2vec safetensors → pytorch_model.bin
    wav2vec_dir        = weights / "wav2vec" / "wav2vec2-base-960h"
    pytorch_weight     = wav2vec_dir / "pytorch_model.bin"
    safetensors_weight = wav2vec_dir / "model.safetensors"
    if safetensors_weight.exists() and not pytorch_weight.exists():
        log("  ⚙  Converting wav2vec safetensors → pytorch_model.bin…")
        try:
            from safetensors.torch import load_file
            import torch
            torch.save(load_file(str(safetensors_weight)), str(pytorch_weight))
            log("  ✓  Conversion done")
        except Exception as e:
            log(f"  ⚠  Conversion failed: {e}")

    # /app/pretrained_models → HALLO_WEIGHTS
    app_pre = Path("/app/pretrained_models")
    if not app_pre.is_symlink():
        if app_pre.exists():
            app_pre.rename("/app/pretrained_models.bak")
        app_pre.symlink_to(weights.resolve())
    log("  ✓  Model readiness check complete")


log("⏳ Running model readiness check…")
ensure_models_ready()


def make_square_image(input_path: str, output_path: str) -> str:
    """Center-crop to square — prevents face distortion from Resize(512,512)"""
    try:
        from PIL import Image
        img = Image.open(input_path).convert("RGB")
        w, h = img.size
        log(f"  📐 Input image: {w}×{h}")
        if w != h:
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
            img = img.resize((512, 512), Image.LANCZOS)
            img.save(output_path, "JPEG", quality=95)
            log(f"  ✓  Cropped to 512×512 square")
            return output_path
        else:
            img.resize((512, 512), Image.LANCZOS).save(output_path, "JPEG", quality=95)
            log(f"  ✓  Already square, resized to 512×512")
            return output_path
    except Exception as e:
        log(f"  ⚠  Square crop failed ({e}) — using original")
        return input_path


def write_job_config(image_path, audio_path, job_save_dir,
                     pose_weight, face_weight, lip_weight,
                     face_expand_ratio, steps, tmp_dir) -> str:
    base_cfg = "/app/configs/inference/default.yaml"
    if not os.path.exists(base_cfg):
        raise FileNotFoundError(f"Missing: {base_cfg}")

    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)

    W = HALLO_WEIGHTS

    cfg["source_image"]       = image_path
    cfg["driving_audio"]      = audio_path
    # Each job gets its own unique save_path directory
    # inference.py writes: save_path/output.mp4
    cfg["save_path"]          = str(job_save_dir)
    cfg["audio_ckpt_dir"]     = f"{W}/hallo"
    cfg["base_model_path"]    = f"{W}/stable-diffusion-v1-5"
    cfg["motion_module_path"] = f"{W}/motion_module/mm_sd_v15_v2.ckpt"

    if not isinstance(cfg.get("face_analysis"), dict):
        cfg["face_analysis"] = {}
    cfg["face_analysis"]["model_path"] = f"{W}/face_analysis"

    if not isinstance(cfg.get("wav2vec"), dict):
        cfg["wav2vec"] = {}
    cfg["wav2vec"]["model_path"] = f"{W}/wav2vec/wav2vec2-base-960h"
    cfg["wav2vec"].setdefault("features", "all")

    if not isinstance(cfg.get("vae"), dict):
        cfg["vae"] = {}
    cfg["vae"]["model_path"] = f"{W}/sd-vae-ft-mse"

    if not isinstance(cfg.get("audio_separator"), dict):
        cfg["audio_separator"] = {}
    cfg["audio_separator"]["model_path"] = f"{W}/audio_separator/Kim_Vocal_2.onnx"

    cfg["inference_steps"]   = steps
    cfg["pose_weight"]       = pose_weight
    cfg["face_weight"]       = face_weight
    cfg["lip_weight"]        = lip_weight
    cfg["face_expand_ratio"] = face_expand_ratio

    cfg_path = str(tmp_dir / "job_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    log(f"  save_path (unique per job) : {job_save_dir}")
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

        # Each job has its own isolated output directory — no concurrent collision
        job_save_dir = tmp / "output"
        job_save_dir.mkdir()

        # Download image
        image_url = inp.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required", "job_id": job_id}
        img_ext      = Path(image_url.split("?")[0]).suffix or ".jpg"
        raw_img_path = str(tmp / f"raw{img_ext}")
        sq_img_path  = str(tmp / "portrait_512.jpg")
        try:
            r = requests.get(image_url, stream=True, timeout=120)
            r.raise_for_status()
            with open(raw_img_path, "wb") as f:
                for chunk in r.iter_content(65536): f.write(chunk)
            log(f"     ✓ {os.path.getsize(raw_img_path)/1e6:.1f} MB")
        except Exception as e:
            return {"error": f"Image download failed: {e}", "job_id": job_id}

        image_path = make_square_image(raw_img_path, sq_img_path)

        # Download audio
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

        pose_weight       = float(inp.get("pose_weight",       1.0))
        face_weight       = float(inp.get("face_weight",       1.0))
        lip_weight        = float(inp.get("lip_weight",        1.0))
        face_expand_ratio = float(inp.get("face_expand_ratio", 1.2))
        steps             = int(inp.get("steps",               40))

        log(f"  pose={pose_weight}  face={face_weight}  lip={lip_weight}  "
            f"expand={face_expand_ratio}  steps={steps}")

        try:
            cfg_path = write_job_config(
                image_path, audio_path, job_save_dir,
                pose_weight, face_weight, lip_weight,
                face_expand_ratio, steps, tmp)
        except Exception as e:
            return {"error": f"Config failed: {e}", "job_id": job_id}

        cmd = [sys.executable, "scripts/inference.py", "--config", cfg_path]
        log(f"  Command: {' '.join(cmd)}\n")
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

        # Find output — Hallo writes save_path/output.mp4
        video_path = None
        expected = job_save_dir / "output.mp4"
        if expected.exists() and expected.stat().st_size > 0:
            video_path = str(expected)
        else:
            # Glob fallback
            for p in sorted(job_save_dir.glob("*.mp4")):
                if p.stat().st_size > 0:
                    video_path = str(p)
                    break

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
                "dir":    [str(p) for p in job_save_dir.iterdir()],
            }

        log(f"  ✓ Output: {video_path}")
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
