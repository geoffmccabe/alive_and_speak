"""
RunPod Serverless Handler – Hallo
Dynamic Path Symlink-Safe Version
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
import shutil
from pathlib import Path
from typing import Union

import runpod

# ─────────────────────────────────────────────────────────────────
# 🧭 DYNAMIC PATH & ENVIRONMENT RESOLUTION (Pod vs Serverless)
# ─────────────────────────────────────────────────────────────────
def resolve_hallo_weights_root() -> str:
    """
    Safely resolves the path whether running in an interactive Pod or Serverless,
    accounting for RunPod's workspace-to-volume symlinking structure.
    """
    if os.environ.get("HALLO_WEIGHTS"):
        return os.environ.get("HALLO_WEIGHTS")
        
    serverless_path = Path("/runpod-volume/weights/hallo")
    pod_path = Path("/workspace/weights/hallo")
    
    # Resolve the real path to bypass symlink collisions
    if serverless_path.exists():
        print(f"🧭 [Auto-Detect] Serverless environment: {serverless_path}", flush=True)
        return str(serverless_path.resolve())
    elif pod_path.exists():
        print(f"🧭 [Auto-Detect] Interactive Pod environment: {pod_path}", flush=True)
        return str(pod_path.resolve())
        
    return "/runpod-volume/weights/hallo"

HALLO_WEIGHTS      = resolve_hallo_weights_root()
OUTPUT_DIR         = "/tmp/hallo_outputs"
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "600"))

# Enforce strict offline configurations
os.environ["HF_HOME"]               = HALLO_WEIGHTS
os.environ["TRANSFORMERS_CACHE"]    = HALLO_WEIGHTS
os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["HF_DATASETS_OFFLINE"]   = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"

def log(msg: str) -> None:
    print(msg, flush=True)

def safe_mkdir(target_path: Path):
    """Symlink-safe directory creation to prevent Errno 17 FileExistsError."""
    if not target_path.exists():
        try:
            os.makedirs(str(target_path), exist_ok=True)
        except FileExistsError:
            pass

# ─────────────────────────────────────────────────────────────────
# 🎯 AUTOMATED RUNTIME ASSET REPAIR LAYER
# ─────────────────────────────────────────────────────────────────
def initialize_and_enforce_assets():
    log(f"⏳ Initiating layout validation using base path target: {HALLO_WEIGHTS}")
    
    models_dir = Path(HALLO_WEIGHTS) / "face_analysis/models"
    buffalo_dir = models_dir / "buffalo_l"
    wav2vec_dir = Path(HALLO_WEIGHTS) / "wav2vec/wav2vec2-base-960h"
    
    safe_mkdir(models_dir)
    safe_mkdir(buffalo_dir)
    safe_mkdir(wav2vec_dir)

    # 🛠️ SAFETENSORS TO PYTORCH_MODEL.BIN CONVERSION
    safetensors_weight = wav2vec_dir / "model.safetensors"
    pytorch_weight = wav2vec_dir / "pytorch_model.bin"
    
    if safetensors_weight.exists() and not pytorch_weight.exists():
        log("   ⚙️ Found 'model.safetensors' for wav2vec. Converting to required 'pytorch_model.bin' layout...")
        try:
            from safetensors.torch import load_file
            import torch
            tensors = load_file(str(safetensors_weight))
            torch.save(tensors, str(pytorch_weight))
            log("     ✓ Safetensors conversion successful!")
        except Exception as conv_err:
            log(f"     ⚠️ Automatic structural weight translation failed: {conv_err}")

    # Core checklist for download layouts if missing
    download_targets = [
        (
            "https://huggingface.co/fudan-generative-ai/hallo2/resolve/main/face_analysis/models/face_landmarker_v2_with_blendshapes.task",
            models_dir / "face_landmarker_v2_with_blendshapes.task"
        ),
        (
            "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/1k3d68.onnx",
            buffalo_dir / "1k3d68.onnx"
        ),
        (
            "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
            buffalo_dir / "det_10g.onnx"
        ),
        (
            "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
            buffalo_dir / "w600k_r50.onnx"
        )
    ]

    for url, dest_path in download_targets:
        if not dest_path.exists() or dest_path.stat().st_size == 0:
            log(f"   ✗ Missing target configuration: {dest_path.name}. Repairing alignment...")
            try:
                os.environ["HF_HUB_OFFLINE"] = "0"
                with requests.get(url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(dest_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                log(f"     ✓ Fixed asset alignment layout: {dest_path.name}")
            except Exception as e:
                log(f"   ⚠️ Layout establishment hook failed: {e}")
            finally:
                os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            log(f"   ✓ Verified asset presence: {dest_path.name}")

    # Align ONNX flat files for InsightFace wrappers
    for onnx_file in buffalo_dir.glob("*.onnx"):
        flat_link_target = models_dir / onnx_file.name
        if not flat_link_target.exists() and not flat_link_target.is_symlink():
            try:
                flat_link_target.symlink_to(onnx_file)
            except Exception:
                pass

    # Anchor target paths inside execution environments cleanly
    for base_prefix in ["/app", ""]:
        target_models_dir = f"{base_prefix}/pretrained_models/face_analysis/models"
        try:
            if os.path.exists(target_models_dir) or os.path.islink(target_models_dir):
                if os.path.islink(target_models_dir):
                    os.unlink(target_models_dir)
                else:
                    shutil.rmtree(target_models_dir)
            os.makedirs(os.path.dirname(target_models_dir), exist_ok=True)
            os.symlink(str(models_dir), target_models_dir)
        except Exception:
            pass

# Execute verification before running worker
initialize_and_enforce_assets()
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────
def download_file(url: str, dest: str, timeout: int = 120) -> str:
    log(f"  ⬇  {url}")
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65_536):
                fh.write(chunk)
        log(f"     ✓ {os.path.getsize(dest)/1e6:.1f} MB")
    finally:
        os.environ["HF_HUB_OFFLINE"] = "1"
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
        raise FileNotFoundError(f"Base config template missing: {base_cfg}")

    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)

    cfg["source_image"]  = image_path
    cfg["driving_audio"] = audio_path
    cfg["save_path"]     = output_path

    cfg["base_model_path"]    = f"{HALLO_WEIGHTS}/stable-diffusion-v1-5"
    cfg["motion_module_path"] = f"{HALLO_WEIGHTS}/motion_module/mm_sd_v15_v2.ckpt"
    cfg["audio_ckpt_dir"]     = f"{HALLO_WEIGHTS}/hallo"

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

    cfg["inference_steps"]   = steps
    cfg["pose_weight"]       = pose_weight
    cfg["face_weight"]       = face_weight
    cfg["lip_weight"]        = lip_weight
    cfg["face_expand_ratio"] = face_expand_ratio

    cfg_path = str(tmp_dir / "job_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    return cfg_path


# ─────────────────────────────────────────────────────────────────
#  Handler Processing Loop
# ─────────────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    job_id = job["id"]
    inp    = job["input"]

    log(f"\n{'═'*60}")
    log(f"  Hallo Job : {job_id}")
    log(f"  Dynamic Weight Target Location: {HALLO_WEIGHTS}")
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

        pose_weight        = float(inp.get("pose_weight",        1.0))
        face_weight        = float(inp.get("face_weight",        1.0))
        lip_weight         = float(inp.get("lip_weight",         1.0))
        face_expand_ratio  = float(inp.get("face_expand_ratio",  1.2))
        steps              = int(inp.get("steps",                 40))

        output_path = str(Path(OUTPUT_DIR) / f"{job_id}.mp4")

        try:
            cfg_path = write_job_config(
                image_path, audio_path, output_path,
                pose_weight, face_weight, lip_weight,
                face_expand_ratio, steps, tmp)
        except Exception as e:
            return {"error": f"Config compilation failed: {e}", "job_id": job_id}

        cmd = [sys.executable, "scripts/inference.py", "--config", cfg_path]

        try:
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            cuda_paths = "/usr/local/cuda/lib64:/usr/local/cuda-12.1/lib64"
            new_ld = f"{cuda_paths}:{current_ld}" if current_ld else cuda_paths

            custom_env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "INSIGHTFACE_HOME": f"{HALLO_WEIGHTS}/face_analysis",
                "LD_LIBRARY_PATH": new_ld,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1"
            }

            proc = subprocess.Popen(
                cmd, cwd="/app",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env=custom_env,
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
            return {"error": f"Inference execution failed: {e}", "job_id": job_id}

        if return_code != 0:
            log_output = "\n".join(lines[-60:])
            return {
                "error":      "Pipeline processing failed",
                "returncode": return_code,
                "output":     log_output,
                "job_id":     job_id,
            }

        video_path = find_output_video(OUTPUT_DIR, job_id)
        if not video_path:
            return {
                "error":  "Rendered output missing from tmp directory",
                "job_id": job_id,
                "dir":    [str(p) for p in Path(OUTPUT_DIR).iterdir()],
            }

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
            return {"error": f"Response coding failed: {e}", "job_id": job_id}


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    runpod.serverless.start({"handler": handler})
