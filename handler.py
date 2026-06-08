"""
RunPod Serverless Handler – Hallo
https://github.com/fudan-generative-vision/hallo

Key fixes:
  1. Input image is center-cropped to square before inference.
     Hallo uses transforms.Resize(512,512) with no aspect-ratio handling —
     a non-square image gets squashed, causing face/teeth distortion.
  2. save_path must be a DIRECTORY (Hallo calls os.makedirs on it).
     config.output is the actual file path (default: .cache/output.mp4).
  3. /app/.cache must exist for the default output path.
  4. /app/pretrained_models → HALLO_WEIGHTS for relative config paths.
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


def download_file(url: str, dest: str, timeout: int = 180) -> bool:
    try:
        log(f"  ⬇  {url}")
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65_536):
                fh.write(chunk)
        log(f"     ✓ {os.path.getsize(dest)/1e6:.1f} MB → {dest}")
        return True
    except Exception as e:
        log(f"     ✗ Download failed: {e}")
        return False


def ensure_models_ready() -> None:
    weights     = Path(HALLO_WEIGHTS)
    models_dir  = weights / "face_analysis" / "models"
    buffalo_dir = models_dir / "buffalo_l"

    models_dir.mkdir(parents=True, exist_ok=True)
    buffalo_dir.mkdir(parents=True, exist_ok=True)

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
            download_file(url, str(fpath))
        else:
            log(f"  ✓  {fname} ({fpath.stat().st_size/1e6:.1f} MB)")

    linked = 0
    for src in buffalo_dir.glob("*.onnx"):
        dst = models_dir / src.name
        if not dst.exists() and not dst.is_symlink():
            dst.symlink_to(src.resolve())
            linked += 1
    if linked:
        log(f"  ✓  Symlinked {linked} .onnx flat into models/")

    landmarker = models_dir / "face_landmarker_v2_with_blendshapes.task"
    if not landmarker.exists():
        log(f"  ⚠  Missing face_landmarker — downloading…")
        download_file(
            "https://huggingface.co/fudan-generative-ai/hallo2/resolve/main/face_analysis/models/face_landmarker_v2_with_blendshapes.task",
            str(landmarker)
        )

    wav2vec_dir        = weights / "wav2vec" / "wav2vec2-base-960h"
    pytorch_weight     = wav2vec_dir / "pytorch_model.bin"
    safetensors_weight = wav2vec_dir / "model.safetensors"
    if safetensors_weight.exists() and not pytorch_weight.exists():
        log("  ⚙  Converting wav2vec safetensors → pytorch_model.bin…")
        try:
            from safetensors.torch import load_file
            import torch
            tensors = load_file(str(safetensors_weight))
            torch.save(tensors, str(pytorch_weight))
            log("  ✓  Conversion successful")
        except Exception as e:
            log(f"  ⚠  Conversion failed: {e}")

    # /app/pretrained_models → HALLO_WEIGHTS (relative paths in default.yaml)
    app_pretrained = Path("/app/pretrained_models")
    if not app_pretrained.is_symlink():
        if app_pretrained.exists():
            app_pretrained.rename("/app/pretrained_models.bak")
        app_pretrained.symlink_to(weights.resolve())
        log(f"  ✓  Linked /app/pretrained_models → {weights}")
    else:
        log(f"  ✓  /app/pretrained_models linked")

    # /app/.cache — Hallo writes output.mp4 here
    Path("/app/.cache").mkdir(exist_ok=True)
    log(f"  ✓  /app/.cache ready")
    log("  ✓  Model readiness check complete")


log("⏳ Running model readiness check…")
ensure_models_ready()


# ─────────────────────────────────────────────────────────────────
#  Image preprocessing — center-crop to square
#  CRITICAL: Hallo does Resize(512,512) with no aspect ratio handling.
#  A non-square image gets squashed → face distortion, teeth artifacts.
#  We center-crop to square before passing to inference.
# ─────────────────────────────────────────────────────────────────
def make_square_image(input_path: str, output_path: str) -> str:
    try:
        from PIL import Image
        img = Image.open(input_path).convert("RGB")
        w, h = img.size
        log(f"  📐 Input image: {w}×{h}")

        if w == h:
            log(f"  ✓  Already square — no crop needed")
            img.save(output_path, "JPEG", quality=95)
            return output_path

        # Center crop to square
        side = min(w, h)
        left   = (w - side) // 2
        top    = (h - side) // 2
        right  = left + side
        bottom = top  + side
        img_cropped = img.crop((left, top, right, bottom))

        # Resize to 512×512 (Hallo's expected size)
        img_resized = img_cropped.resize((512, 512), Image.LANCZOS)
        img_resized.save(output_path, "JPEG", quality=95)
        log(f"  ✓  Cropped {w}×{h} → {side}×{side} → 512×512")
        return output_path

    except Exception as e:
        log(f"  ⚠  Image preprocessing failed ({e}) — using original")
        return input_path


def download_input(url: str, dest: str) -> str:
    log(f"  ⬇  {url}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=65_536):
            fh.write(chunk)
    log(f"     ✓ {os.path.getsize(dest)/1e6:.1f} MB")
    return dest


def find_output_video(job_id: str) -> Union[str, None]:
    candidates = [
        Path("/app/.cache/output.mp4"),
        Path(OUTPUT_DIR) / f"{job_id}.mp4",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return str(p)
    for p in sorted(Path("/app/.cache").glob("*.mp4")):
        if p.stat().st_size > 0:
            return str(p)
    for p in sorted(Path(OUTPUT_DIR).glob("*.mp4")):
        if p.stat().st_size > 0:
            return str(p)
    return None


def write_job_config(image_path, audio_path,
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
    # save_path must be a DIRECTORY — Hallo calls os.makedirs on it
    # The actual output file is config.output (default: .cache/output.mp4)
    cfg["save_path"]          = "/app/.cache"
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

    log(f"  face_analysis.model_path : {cfg['face_analysis']['model_path']}")
    log(f"  save_path (dir)          : {cfg['save_path']}")
    return cfg_path


def handler(job: dict) -> dict:
    job_id = job["id"]
    inp    = job["input"]

    log(f"\n{'═'*60}")
    log(f"  Hallo Job : {job_id}")
    log(f"  Keys      : {list(inp.keys())}")
    log(f"{'═'*60}\n")

    # Clear leftover output from previous job
    cache_out = Path("/app/.cache/output.mp4")
    if cache_out.exists():
        cache_out.unlink()

    with tempfile.TemporaryDirectory(prefix=f"hallo_{job_id}_") as _tmp:
        tmp = Path(_tmp)

        # ── Download image ────────────────────────────────────────
        image_url = inp.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required", "job_id": job_id}
        img_ext      = Path(image_url.split("?")[0]).suffix or ".jpg"
        raw_img_path = str(tmp / f"raw_portrait{img_ext}")
        sq_img_path  = str(tmp / "portrait_512.jpg")
        try:
            download_input(image_url, raw_img_path)
        except Exception as e:
            return {"error": f"Image download failed: {e}", "job_id": job_id}

        # Center-crop to square — prevents face distortion in Hallo
        image_path = make_square_image(raw_img_path, sq_img_path)

        # ── Download audio ────────────────────────────────────────
        audio_url = inp.get("audio_url")
        if not audio_url:
            return {"error": "'audio_url' is required", "job_id": job_id}
        aud_ext    = Path(audio_url.split("?")[0]).suffix or ".wav"
        audio_path = str(tmp / f"driving_audio{aud_ext}")
        try:
            download_input(audio_url, audio_path)
        except Exception as e:
            return {"error": f"Audio download failed: {e}", "job_id": job_id}

        # ── Parameters ───────────────────────────────────────────
        pose_weight       = float(inp.get("pose_weight",       1.0))
        face_weight       = float(inp.get("face_weight",       1.0))
        lip_weight        = float(inp.get("lip_weight",        1.0))
        face_expand_ratio = float(inp.get("face_expand_ratio", 1.2))
        steps             = int(inp.get("steps",               40))

        log(f"  pose={pose_weight}  face={face_weight}  lip={lip_weight}  "
            f"expand={face_expand_ratio}  steps={steps}")

        # ── Config ───────────────────────────────────────────────
        try:
            cfg_path = write_job_config(
                image_path, audio_path,
                pose_weight, face_weight, lip_weight,
                face_expand_ratio, steps, tmp)
        except Exception as e:
            return {"error": f"Config failed: {e}", "job_id": job_id}

        # ── Run inference ────────────────────────────────────────
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

        video_path = find_output_video(job_id)

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
                "cache":  [str(p) for p in Path("/app/.cache").iterdir()],
            }

        log(f"  ✓ Output: {video_path}")
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
