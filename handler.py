"""
RunPod Serverless Handler – MultiTalk (alive_and_speak)
Upstream https://github.com/MeiGen-AI/MultiTalk
"""

import os
import sys
import json
import time
import requests
import tempfile
import subprocess
from pathlib import Path
import runpod

# ─────────────────────────────────────────────────────────────────
#  Model Paths & Environment Configuration
# ─────────────────────────────────────────────────────────────────
WEIGHTS_ROOT  = os.environ.get("WEIGHTS_ROOT",  "/runpod-volume/weights")
CKPT_DIR      = os.environ.get("CKPT_DIR",      f"{WEIGHTS_ROOT}/Wan2.1-I2V-14B-480P")
WAV2VEC_DIR   = os.environ.get("WAV2VEC_DIR",   f"{WEIGHTS_ROOT}/chinese-wav2vec2-base")
KOKORO_DIR    = os.environ.get("KOKORO_DIR",    f"{WEIGHTS_ROOT}/Kokoro-82M")
MULTITALK_DIR = os.environ.get("MULTITALK_DIR", f"{WEIGHTS_ROOT}/MeiGen-MultiTalk")

OUTPUT_DIR         = os.environ.get("OUTPUT_DIR",         "/tmp/multitalk_outputs")
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "18000"))

# ─────────────────────────────────────────────────────────────────
#  Supabase Upload Configuration
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
#  Helpers & Hotfixes
# ─────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(msg, flush=True)


def apply_runtime_hotfixes() -> None:
    """
    Safely applies hotfixes to codebase scripts and configurations to bypass
    Hugging Face SDPA/Eager attention conflicts during runtime.
    """
    log("🔧 Applying runtime environment hotfixes...")

    script_path = "/app/src/audio_analysis/wav2vec2.py"
    if os.path.exists(script_path):
        try:
            with open(script_path, "r") as f:
                code = f.read()
            if 'attn_implementation = "eager"' not in code:
                old_stmt = "self.config.output_attentions = True"
                new_stmt = 'self.config.attn_implementation = "eager"\n        self.config.output_attentions = True'
                if old_stmt in code:
                    code = code.replace(old_stmt, new_stmt)
                    with open(script_path, "w") as f:
                        f.write(code)
                    log("   ✓ Code patch successfully applied to wav2vec2.py")
                else:
                    log("   ⚠ Target statement not found in wav2vec2.py")
            else:
                log("   ✓ wav2vec2.py script already has hotfix applied.")
        except Exception as e:
            log(f"   ⚠ Failed to patch script file: {e}")

    config_path = os.path.join(WAV2VEC_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            if cfg.get("attn_implementation") != "eager":
                cfg["attn_implementation"] = "eager"
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                log("   ✓ Model config.json patched to force 'eager' attention.")
            else:
                log("   ✓ Model config.json is already configured correctly.")
        except Exception as e:
            log(f"   ⚠ Failed to update configuration file: {e}")


def download_file(url: str, dest: str, timeout: int = 180) -> str:
    log(f"  ⬇  {url}")
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=65_536):
            fh.write(chunk)
    mb = os.path.getsize(dest) / 1_000_000
    log(f"     ✓ {mb:.1f} MB → {dest}")
    return dest


def upload_video(video_path: str, job_id: str) -> dict:
    """
    Fetches a pre-signed direct upload URL from the Supabase Edge Function,
    then streams the binary file via PUT directly to the storage bucket.
    This fully resolves HTTP 504 timeouts on massive files.
    """
    log(f"  ⬆  Initiating multi-stage pre-signed upload for {video_path} …")
    mb = os.path.getsize(video_path) / 1_000_000
    log(f"     Size : {mb:.1f} MB")
    filename = f"{job_id}.mp4"

    try:
        # Step 1: Request a direct signed token URL from the Edge Function proxy
        log("     Fetching upload token from Edge Function...")
        token_resp = requests.post(
            SIGNED_UPLOAD_ENDPOINT,
            headers={
                "Authorization": f"Bearer {RUNPOD_UPLOAD_SECRET}",
                "Content-Type":  "application/json",
            },
            json={"filename": filename},
            timeout=30
        )

        if not token_resp.ok:
            return {
                "upload_error":   f"Failed token generation: HTTP {token_resp.status_code}",
                "upload_response": token_resp.text[:1000],
                "video_url":      "",
            }

        urls = token_resp.json()
        upload_url = urls.get("uploadUrl")
        public_url = urls.get("publicUrl")

        if not upload_url or not public_url:
            return {
                "upload_error":   "Malformed token response structure from storage proxy",
                "upload_response": urls,
                "video_url":      "",
            }

        # Step 2: Directly upload the file binary stream into the storage engine via PUT
        log("     Streaming binary payload straight to Supabase Storage Bucket...")
        with open(video_path, "rb") as fh:
            storage_resp = requests.put(
                upload_url,
                headers={"Content-Type": "video/mp4"},
                data=fh,
                timeout=600  # Generous 10-minute timeout for slow file chunk pushes
            )

        log(f"     Storage Engine HTTP {storage_resp.status_code}")

        if storage_resp.ok:
            log(f"     ✓ Upload Complete! Public Access Link: {public_url}")
            return {
                "video_url":       public_url, 
                "upload_response": {"status": "success", "url": public_url}
            }
        else:
            return {
                "upload_error":   f"Storage backend rejected payload: HTTP {storage_resp.status_code}",
                "upload_response": storage_resp.text[:1000],
                "video_url":      "",
            }

    except Exception as e:
        log(f"  ❌ Upload network failure: {str(e)}")
        return {
            "upload_error": "Network/Timeout exception during multi-stage upload sequence",
            "upload_response": str(e),
            "video_url": "",
        }


def find_output_video(save_stem: str) -> str | None:
    for suffix in ["", "_out", "_final", "_0"]:
        p = f"{save_stem}{suffix}.mp4"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    stem_name = Path(save_stem).name
    for p in sorted(Path(OUTPUT_DIR).glob(f"{stem_name}*.mp4")):
        if p.stat().st_size > 0:
            return str(p)
    return None


# ─────────────────────────────────────────────────────────────────
#  Main Handler
# ─────────────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    apply_runtime_hotfixes()

    os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
    os.environ["TORCH_ATTN_IMPLEMENTATION"] = "eager"

    job_id = job["id"]
    inp    = job["input"]

    log(f"\n{'═' * 60}")
    log(f"   Job  : {job_id}")
    log(f"   Keys : {list(inp.keys())}")
    log(f"{'═' * 60}\n")

    with tempfile.TemporaryDirectory(prefix=f"mt_{job_id}_") as _tmp:
        tmp = Path(_tmp)

        # ── 1. Download reference image ───────────────────────────
        image_url = inp.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required", "job_id": job_id}

        img_ext    = Path(image_url.split("?")[0]).suffix or ".jpg"
        image_path = str(tmp / f"ref_image{img_ext}")
        try:
            download_file(image_url, image_path)
        except Exception as exc:
            return {"error": f"Image download failed: {exc}", "job_id": job_id}

        # ── 2. Prepare audio ──────────────────────────────────────
        audio_mode = inp.get("audio_mode", "audio")

        # ── 3. Build input JSON for the backend script ────────────
        prompt     = inp.get("prompt", "A person talking naturally and expressively")
        neg_prompt = inp.get("negative_prompt", "distorted face, blurry, low quality")

        gen_input: dict = {
            "prompt":          prompt,
            "negative_prompt": neg_prompt,
            "cond_image":      image_path,
        }

        if audio_mode == "audio":
            audio_urls = inp.get("audio_urls") or inp.get("audio_url")
            if not audio_urls:
                return {"error": "'audio_urls' required when audio_mode='audio'", "job_id": job_id}
            if isinstance(audio_urls, str):
                audio_urls = [audio_urls]

            audio_paths: list[str] = []
            for i, url in enumerate(audio_urls[:2]):
                a_ext  = Path(url.split("?")[0]).suffix or ".wav"
                a_path = str(tmp / f"audio_{i}{a_ext}")
                try:
                    download_file(url, a_path)
                    audio_paths.append(a_path)
                except Exception as exc:
                    return {"error": f"Audio[{i}] download failed: {exc}", "job_id": job_id}

            if len(audio_paths) == 1:
                gen_input["cond_audio"] = {"person1": audio_paths[0]}
            else:
                gen_input["cond_audio"] = {
                    "person1": audio_paths[0],
                    "person2": audio_paths[1],
                }
                gen_input["audio_type"] = inp.get("audio_type", "para")

        elif audio_mode == "tts":
            tts_texts  = inp.get("tts_texts", [])
            tts_voices = inp.get("tts_voices", [])
            if not tts_texts:
                return {"error": "'tts_texts' required when audio_mode='tts'", "job_id": job_id}
            if isinstance(tts_texts, str):
                tts_texts = [tts_texts]

            voice_defaults = ["af_heart", "am_adam"]
            resolved_voices = []
            for i in range(len(tts_texts[:2])):
                voice_name = tts_voices[i] if i < len(tts_voices) else voice_defaults[min(i, 1)]
                if voice_name.startswith("/") or voice_name.startswith("."):
                    resolved_voices.append(voice_name)
                else:
                    if not voice_name.endswith(".pt") and not voice_name.endswith(".pth"):
                        voice_name = f"{voice_name}.pt"
                    resolved_voices.append(os.path.join(KOKORO_DIR, "voices", voice_name))

            if len(tts_texts) == 1:
                gen_input["tts_audio"] = {
                    "text":         tts_texts[0],
                    "human1_voice": resolved_voices[0],
                }
                gen_input["cond_audio"] = {"person1": ""}
            else:
                gen_input["tts_audio"] = {
                    "text1":        tts_texts[0],
                    "text2":        tts_texts[1],
                    "human1_voice": resolved_voices[0],
                    "human2_voice": resolved_voices[1],
                }
                gen_input["cond_audio"] = {"person1": "", "person2": ""}
                gen_input["audio_type"] = inp.get("audio_type", "para")
        else:
            return {"error": f"Unknown audio_mode '{audio_mode}'", "job_id": job_id}

        input_json = str(tmp / "input.json")
        with open(input_json, "w") as fh:
            json.dump(gen_input, fh, indent=2)

        # ── 4. Parameter Gathering ──────────────────────────────
        save_stem       = str(Path(OUTPUT_DIR) / job_id)
        audio_save_dir  = str(tmp / "save_audio")
        os.makedirs(audio_save_dir, exist_ok=True)

        mode            = inp.get("mode", "streaming")
        sample_steps    = int(inp.get("sample_steps", 15))
        size            = inp.get("size", "multitalk-480")
        use_teacache    = bool(inp.get("use_teacache", True))
        use_apg         = bool(inp.get("use_apg", False))
        num_persistent  = int(inp.get("num_persistent_param_in_dit", 0))
        txt_guide       = float(inp.get("sample_text_guide_scale", 5.0))
        aud_guide       = float(inp.get("sample_audio_guide_scale", 4.0))
        teacache_thresh = float(inp.get("teacache_thresh", 0.3))
        sample_shift    = inp.get("sample_shift")
        lora_dir        = inp.get("lora_dir") or os.environ.get("LORA_DIR", "")
        lora_scale      = float(inp.get("lora_scale", 1.0))
        quant           = inp.get("quant", "")
        quant_dir       = inp.get("quant_dir") or MULTITALK_DIR
        offload_choice  = inp.get("offload_model", True)

        # ── 5. Build Command List ─────────────────────────────────
        cmd: list[str] = [
            sys.executable,                    "/app/generate_multitalk.py",
            "--ckpt_dir",                      CKPT_DIR,
            "--wav2vec_dir",                   WAV2VEC_DIR,
            "--input_json",                    input_json,
            "--audio_save_dir",                audio_save_dir,
            "--sample_steps",                  str(sample_steps),
            "--mode",                          mode,
            "--size",                          size,
            "--num_persistent_param_in_dit",   str(num_persistent),
            "--sample_text_guide_scale",       str(txt_guide),
            "--sample_audio_guide_scale",      str(aud_guide),
            "--teacache_thresh",               str(teacache_thresh),
            "--save_file",                     save_stem,
        ]

        if offload_choice is False or str(offload_choice).lower() == "false":
            cmd.extend(["--offload_model", "False"])
        else:
            cmd.extend(["--offload_model", "True"])

        if use_teacache:
            cmd.append("--use_teacache")
        if use_apg:
            cmd.append("--use_apg")
        if audio_mode == "tts":
            cmd += ["--audio_mode", "tts"]
        if lora_dir:
            cmd += ["--lora_dir", lora_dir, "--lora_scale", str(lora_scale)]
        if quant:
            cmd += ["--quant", quant, "--quant_dir", quant_dir]
        if sample_shift is not None:
            cmd += ["--sample_shift", str(sample_shift)]

        log(f"  Command:\n  {' '.join(cmd)}\n")

        # ── 6. Run Subprocess with Real-Time Log Streaming ────────
        subprocess_env = os.environ.copy()
        subprocess_env["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
        subprocess_env["TORCH_ATTN_IMPLEMENTATION"] = "eager"
        subprocess_env["PYTHONUNBUFFERED"] = "1"

        start_time = time.time()
        output_lines = []

        log("🚀 Launching generation subprocess...")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd="/app",
                env=subprocess_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            while True:
                if time.time() - start_time > GENERATION_TIMEOUT:
                    log(f"❌ Subprocess exceeded GENERATION_TIMEOUT of {GENERATION_TIMEOUT}s. Terminating...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return {"error": f"Generation timed out after {GENERATION_TIMEOUT}s", "job_id": job_id}

                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    log(line.rstrip("\r\n"))
                    output_lines.append(line)

            return_code = proc.wait()

        except Exception as e:
            return {"error": f"Subprocess execution failed: {str(e)}", "job_id": job_id}

        log(f"  Return code: {return_code}")
        full_output = "".join(output_lines)

        if return_code != 0:
            return {
                "error":      "Video generation failed",
                "returncode": return_code,
                "stderr":     full_output[-4000:],
                "stdout":     full_output[-2000:],
                "job_id":     job_id,
            }

        # ── 7. Find Output Video ──────────────────────────────────
        video_path = find_output_video(save_stem)
        if not video_path:
            return {
                "error":               "Output video not found after generation",
                "save_stem":           save_stem,
                "output_dir_contents": [str(p) for p in Path(OUTPUT_DIR).iterdir()],
                "stdout_tail":         full_output[-500:],
                "job_id":              job_id,
            }

        log(f"  ✓ Found Output Video Local Path: {video_path}")

        # ── 8. Upload to Supabase Storage ─────────────────────────
        upload_result = upload_video(video_path, job_id)

        # Retain local file inside OUTPUT_DIR if upload fails for manual debug extraction
        if upload_result.get("upload_error") or not upload_result.get("video_url"):
            return {
                "status": "upload_failed",
                "job_id": job_id,
                **upload_result
            }

        # Success path cleanup: delete local file to reclaim storage space
        for p in Path(OUTPUT_DIR).glob(f"{job_id}*"):
            try:
                p.unlink()
            except Exception:
                pass

        return {
            "status":          "success",
            "job_id":          job_id,
            "video_url":       upload_result["video_url"],
            "upload_response": upload_result["upload_response"]
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
