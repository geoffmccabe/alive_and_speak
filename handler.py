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

# ─────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────
HALLO_WEIGHTS      = os.environ.get("HALLO_WEIGHTS",      "/runpod-volume/weights/hallo")
OUTPUT_DIR         = os.environ.get("OUTPUT_DIR",         "/tmp/hallo_outputs")
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "600"))  # 10 min

# Point HF cache and general caches at the network volume so nothing downloads at runtime
os.environ["HF_HOME"]            = HALLO_WEIGHTS
os.environ["TRANSFORMERS_CACHE"] = HALLO_WEIGHTS

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


def find_output_video(save_dir: str, job_id: str) -> Union[str, None]:
    exact = Path(save_dir) / f"{job_id}.mp4"
    if exact.exists() and exact.stat().st_size > 0:
        return str(exact)
    for p in sorted(Path(save_dir).glob("*.mp4")):
        if p.stat().st_size > 0 and job_id[:8] in p.name:
            return str(p)
    return None


def build_hallo_config(
    image_path: str,
    audio_path: str,
    output_path: str,
    pose_weight: float,
    face_weight: float,
    lip_weight: float,
    face_expand_ratio: float,
    steps: int,
    tmp_dir: Path,
) -> str:
    """
    Loads Hallo's native default configuration file, modifies the necessary 
    fields to point to our input assets and storage targets, and fixes the 
    InsightFace model layout constraints.
    """
    base_config_path = "/app/configs/inference/default.yaml"
    
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config not found at {base_config_path}")
        
    with open(base_config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # 1. Map runtime inputs onto the template
    config_data["source_image"] = image_path
    config_data["driving_audio"] = audio_path
    config_data["save_path"] = output_path

    # 2. Map weight directories to the persistent network volume assets
    config_data["base_model_path"] = f"{HALLO_WEIGHTS}/stable-diffusion-v1-5"
    config_data["motion_module_path"] = f"{HALLO_WEIGHTS}/motion_module/mm_sd_v15_v2.ckpt"
    config_data["vae_model_path"] = f"{HALLO_WEIGHTS}/sd-vae-ft-mse"
    config_data["ckpt_path"] = f"{HALLO_WEIGHTS}/hallo/net.pth"
    config_data["audio_ckpt_dir"] = f"{HALLO_WEIGHTS}/wav2vec/wav2vec2-base-960h"

    # 3. CRITICAL PATCH: Override path configuration parameters
    # Point directly to the parent folder directory where 'models/buffalo_l' resides.
    config_data["face_analysis_model_path"] = f"{HALLO_WEIGHTS}/face_analysis"
    
    # Force 'buffalo_l' configuration strings to override native blank default rules
    if "face_analysis" not in config_data or not config_data["face_analysis"]:
        config_data["face_analysis"] = {}
    
    if isinstance(config_data["face_analysis"], dict):
        config_data["face_analysis"]["model_name"] = "buffalo_l"
        config_data["face_analysis"]["name"] = "buffalo_l"

    # 4. Update execution hyperparameters
    config_data["inference_steps"] = steps
    config_data["pose_weight"] = pose_weight
    config_data["face_weight"] = face_weight
    config_data["lip_weight"] = lip_weight
    config_data["face_expand_ratio"] = face_expand_ratio

    # Save the complete runtime config back out as a temporary YAML file
    cfg_path = str(tmp_dir / "job_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config_data, f)
        
    return cfg_path


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

        # ── 1. Download portrait image ────────────────────────────
        image_url = inp.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required", "job_id": job_id}

        img_ext    = Path(image_url.split("?")[0]).suffix or ".jpg"
        image_path = str(tmp / f"source_portrait{img_ext}")
        try:
            download_file(image_url, image_path)
        except Exception as exc:
            return {"error": f"Image download failed: {exc}", "job_id": job_id}

        # ── 2. Download driving audio ─────────────────────────────
        audio_url = inp.get("audio_url")
        if not audio_url:
            return {"error": "'audio_url' is required", "job_id": job_id}

        aud_ext    = Path(audio_url.split("?")[0]).suffix or ".wav"
        audio_path = str(tmp / f"driving_audio{aud_ext}")
        try:
            download_file(audio_url, audio_path)
        except Exception as exc:
            return {"error": f"Audio download failed: {exc}", "job_id": job_id}

        # ── 3. Parameters ─────────────────────────────────────────
        pose_weight        = float(inp.get("pose_weight",        1.0))
        face_weight        = float(inp.get("face_weight",        1.0))
        lip_weight         = float(inp.get("lip_weight",         1.0))
        face_expand_ratio  = float(inp.get("face_expand_ratio",  1.2))
        steps              = int(inp.get("steps",                 40))

        log(f"  pose_weight       : {pose_weight}")
        log(f"  face_weight       : {face_weight}")
        log(f"  lip_weight        : {lip_weight}")
        log(f"  face_expand_ratio : {face_expand_ratio}")
        log(f"  steps             : {steps}")

        # ── 4. Output path ────────────────────────────────────────
        output_path = str(Path(OUTPUT_DIR) / f"{job_id}.mp4")

        # ── 5. Build modified YAML config ─────────────────────────
        try:
            cfg_path = build_hallo_config(
                image_path, audio_path, output_path,
                pose_weight, face_weight, lip_weight,
                face_expand_ratio, steps, tmp,
            )
        except Exception as config_err:
            return {"error": f"Config compilation failed: {config_err}", "job_id": job_id}

        # ── 6. Build the inference command ────────────────────────
        cmd: list[str] = [
            sys.executable, "scripts/inference.py",
            "--config", cfg_path,
        ]

        log(f"\n  Command:\n  {' '.join(cmd)}\n")
        log(f"  Config path: {cfg_path}\n")

        # ── 7. Run inference ──────────────────────────────────────
        log("  🚀 Launching Hallo diffusion pipeline…\n")
        try:
            # Enforce cache environments and target home path flags explicitly 
            custom_env = {
                **os.environ, 
                "PYTHONUNBUFFERED": "1",
                "HF_HOME": HALLO_WEIGHTS,
                "TRANSFORMERS_CACHE": HALLO_WEIGHTS,
                "INSIGHTFACE_HOME": f"{HALLO_WEIGHTS}/face_analysis"
            }
            
            proc = subprocess.Popen(
                cmd,
                cwd="/app",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=custom_env,
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
                    return {
                        "error":   f"Timed out after {GENERATION_TIMEOUT}s",
                        "job_id":  job_id,
                    }

            return_code = proc.wait()

        except Exception as exc:
            return {"error": f"Subprocess failed: {exc}", "job_id": job_id}

        log(f"\n  Return code: {return_code}")

        if return_code != 0:
            return {
                "error":      "Video synthesis failed — see output for details",
                "returncode": return_code,
                "output":     "\n".join(lines[-60:]),
                "job_id":     job_id,
            }

        # ── 8. Locate output video ────────────────────────────────
        video_path = find_output_video(OUTPUT_DIR, job_id)
        if not video_path:
            return {
                "error":               "Output video not found after successful run",
                "expected_path":       output_path,
                "output_dir_contents": [str(p) for p in Path(OUTPUT_DIR).iterdir()],
                "job_id":              job_id,
            }

        log(f"  ✓ Output: {video_path}")

        # ── 9. Encode to Base64 and return ────────────────────────
        mb = os.path.getsize(video_path) / 1_000_000
        log(f"  ⚙  Encoding {mb:.1f} MB video to Base64…")

        try:
            with open(video_path, "rb") as f:
                b64_string = base64.b64encode(f.read()).decode("utf-8")

            try:
                os.remove(video_path)
            except Exception:
                pass

            return {
                "status":         "success",
                "job_id":         job_id,
                "video_base64":   b64_string,
                "video_filename": f"{job_id}.mp4",
            }
        except Exception as err:
            return {
                "error":   f"Base64 encoding failed: {err}",
                "job_id":  job_id,
            }


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    runpod.serverless.start({"handler": handler})
