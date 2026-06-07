"""
RunPod Serverless Handler – Hallo
https://github.com/fudan-generative-vision/hallo

Path resolution (from reading inference.py + insightface source):
  - inference.py runs cwd=/app
  - InsightFace calls FaceAnalysis(name="", root=face_analysis_model_path)
  - ensure_available resolves dir_path = root/models/  (name="" → flat)
  - glob(root/models/*.onnx) — onnx files must be FLAT in models/, not in buffalo_l/
  - get_model receives full abs path, asserts osp.exists(model_file)
  - The config face_analysis.model_path is passed as `root`
  - So: root/models/*.onnx must all exist as real files or symlinks

  Fix: at startup, symlink buffalo_l/*.onnx → models/*.onnx (flat)
  Also: download any missing onnx files directly from HuggingFace
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


# ─────────────────────────────────────────────────────────────────
#  Model readiness — runs once at startup
# ─────────────────────────────────────────────────────────────────
def ensure_models_ready() -> None:
    """
    1. Ensure all buffalo_l onnx files exist (download if missing)
    2. Symlink buffalo_l/*.onnx flat into models/ (InsightFace needs this)
    3. Convert wav2vec safetensors → pytorch_model.bin if needed
    4. Download face_landmarker task file if missing
    5. Symlink /app/pretrained_models → HALLO_WEIGHTS for relative path resolution
    """
    weights  = Path(HALLO_WEIGHTS)
    models_dir  = weights / "face_analysis" / "models"
    buffalo_dir = models_dir / "buffalo_l"

    models_dir.mkdir(parents=True, exist_ok=True)
    buffalo_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Buffalo_l onnx files ───────────────────────────────────
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

    # ── 2. Symlink buffalo_l/*.onnx flat into models/ ─────────────
    # InsightFace with name="" globs models/*.onnx directly (not in subdir)
    linked = 0
    for src in buffalo_dir.glob("*.onnx"):
        dst = models_dir / src.name
        if not dst.exists() and not dst.is_symlink():
            dst.symlink_to(src.resolve())
            linked += 1
    if linked:
        log(f"  ✓  Symlinked {linked} .onnx files flat into models/")
    else:
        log(f"  ✓  models/ flat layout already in place")

    # ── 3. face_landmarker task file ──────────────────────────────
    landmarker = models_dir / "face_landmarker_v2_with_blendshapes.task"
    if not landmarker.exists():
        log(f"  ⚠  Missing face_landmarker — downloading…")
        download_file(
            "https://huggingface.co/fudan-generative-ai/hallo2/resolve/main/face_analysis/models/face_landmarker_v2_with_blendshapes.task",
            str(landmarker)
        )

    # ── 4. wav2vec safetensors → pytorch_model.bin ───────────────
    wav2vec_dir       = weights / "wav2vec" / "wav2vec2-base-960h"
    pytorch_weight    = wav2vec_dir / "pytorch_model.bin"
    safetensors_weight= wav2vec_dir / "model.safetensors"

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

    # ── 5. /app/pretrained_models → HALLO_WEIGHTS symlink ────────
    # inference.py cwd=/app, default.yaml uses ./pretrained_models/...
    # This makes all relative paths in the config resolve correctly
    app_pretrained = Path("/app/pretrained_models")
    if app_pretrained.is_symlink():
        current = Path(os.readlink(str(app_pretrained)))
        if current.resolve() != weights.resolve():
            app_pretrained.unlink()
            app_pretrained.symlink_to(weights.resolve())
            log(f"  ✓  Re-linked /app/pretrained_models → {weights}")
        else:
            log(f"  ✓  /app/pretrained_models already correctly linked")
    elif app_pretrained.exists():
        app_pretrained.rename("/app/pretrained_models.bak")
        app_pretrained.symlink_to(weights.resolve())
        log(f"  ✓  Linked /app/pretrained_models → {weights}")
    else:
        app_pretrained.symlink_to(weights.resolve())
        log(f"  ✓  Linked /app/pretrained_models → {weights}")

    log("  ✓  Model readiness check complete")


# Run at import time — before first job
log("⏳ Running model readiness check…")
ensure_models_ready()


# ─────────────────────────────────────────────────────────────────
#  Per-request helpers
# ─────────────────────────────────────────────────────────────────
def download_input(url: str, dest: str) -> str:
    log(f"  ⬇  {url}")
    r = requests.get(url, stream=True, timeout=120)
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

    W = HALLO_WEIGHTS

    # Top-level
    cfg["source_image"]    = image_path
    cfg["driving_audio"]   = audio_path
    cfg["save_path"]       = output_path
    cfg["audio_ckpt_dir"]  = f"{W}/hallo"
    cfg["base_model_path"] = f"{W}/stable-diffusion-v1-5"
    cfg["motion_module_path"] = f"{W}/motion_module/mm_sd_v15_v2.ckpt"

    # Nested keys — must match yaml structure exactly
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
    log(f"  vae.model_path           : {cfg['vae']['model_path']}")
    log(f"  audio_ckpt_dir           : {cfg['audio_ckpt_dir']}")
    log(f"  audio_separator          : {cfg['audio_separator']['model_path']}")
    return cfg_path


# ─────────────────────────────────────────────────────────────────
#  Handler
# ─────────────────────────────────────────────────────────────────
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
            download_input(image_url, image_path)
        except Exception as e:
            return {"error": f"Image download failed: {e}", "job_id": job_id}

        audio_url = inp.get("audio_url")
        if not audio_url:
            return {"error": "'audio_url' is required", "job_id": job_id}
        aud_ext    = Path(audio_url.split("?")[0]).suffix or ".wav"
        audio_path = str(tmp / f"driving_audio{aud_ext}")
        try:
            download_input(audio_url, audio_path)
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
