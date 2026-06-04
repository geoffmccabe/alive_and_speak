"""
RunPod Serverless Handler – MultiTalk (alive_and_speak)
https://github.com/MeiGen-AI/MultiTalk

Upload strategy:
  1. POST to Supabase edge function (original direct upload) — 3 retries
  2. If all retries fail → encode video as base64 and return in response
     so the video is NEVER lost even when Supabase is down.
"""

import os
import sys
import json
import time
import base64
import requests
import tempfile
import subprocess
from pathlib import Path

import runpod

# ─────────────────────────────────────────────────────────────────
#  Model paths
# ─────────────────────────────────────────────────────────────────
WEIGHTS_ROOT  = os.environ.get("WEIGHTS_ROOT",  "/runpod-volume/weights")
CKPT_DIR      = os.environ.get("CKPT_DIR",      f"{WEIGHTS_ROOT}/Wan2.1-I2V-14B-480P")
WAV2VEC_DIR   = os.environ.get("WAV2VEC_DIR",   f"{WEIGHTS_ROOT}/chinese-wav2vec2-base")
KOKORO_DIR    = os.environ.get("KOKORO_DIR",    f"{WEIGHTS_ROOT}/Kokoro-82M")
MULTITALK_DIR = os.environ.get("MULTITALK_DIR", f"{WEIGHTS_ROOT}/MeiGen-MultiTalk")

OUTPUT_DIR         = os.environ.get("OUTPUT_DIR",         "/tmp/multitalk_outputs")
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "18000"))

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


def voice_to_path(voice_name: str) -> str:
    if os.sep in voice_name or voice_name.startswith("."):
        return voice_name
    name = voice_name if voice_name.endswith(".pt") else f"{voice_name}.pt"
    return os.path.join(KOKORO_DIR, "voices", name)


def apply_hotfixes() -> None:
    """Force eager attention to avoid SDPA conflicts."""
    script_path = "/app/src/audio_analysis/wav2vec2.py"
    if os.path.exists(script_path):
        try:
            with open(script_path, "r") as f:
                code = f.read()
            if 'attn_implementation = "eager"' not in code:
                old = "self.config.output_attentions = True"
                new = 'self.config.attn_implementation = "eager"\n        self.config.output_attentions = True'
                if old in code:
                    with open(script_path, "w") as f:
                        f.write(code.replace(old, new))
                    log("   ✓ wav2vec2.py patched")
        except Exception as e:
            log(f"   ⚠ wav2vec2 patch failed: {e}")

    config_path = os.path.join(WAV2VEC_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            if cfg.get("attn_implementation") != "eager":
                cfg["attn_implementation"] = "eager"
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                log("   ✓ config.json patched")
        except Exception as e:
            log(f"   ⚠ config patch failed: {e}")


def download_file(url: str, dest: str, timeout: int = 180) -> str:
    log(f"  ⬇  {url}")
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=65_536):
            fh.write(chunk)
    mb = os.path.getsize(dest) / 1_000_000
    log(f"     ✓ {mb:.1f} MB")
    return dest


def video_to_base64(video_path: str) -> str:
    """Encode video file as base64 string."""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def upload_video(video_path: str, job_id: str) -> dict:
    """
    Upload to Supabase with 3 retries + exponential backoff.
    Falls back to base64 if all attempts fail so video is never lost.
    """
    mb       = os.path.getsize(video_path) / 1_000_000
    filename = f"{job_id}.mp4"

    log(f"  ⬆  Uploading {filename} ({mb:.1f} MB) → {SIGNED_UPLOAD_ENDPOINT}")

    last_error = ""

    for attempt in range(1, 4):   # 3 attempts
        wait = 2 ** attempt       # 2s, 4s, 8s
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
                    timeout=300,   # 5 min per attempt
                )

            log(f"     HTTP {resp.status_code}")

            if resp.ok:
                payload   = resp.json()
                video_url = (
                    payload.get("url")
                    or payload.get("publicUrl")
                    or payload.get("signedUrl")
                    or ""
                )
                log(f"     ✓ Uploaded: {video_url}")
                return {
                    "video_url":      video_url,
                    "upload_response": payload,
                    "upload_method":  "supabase",
                }

            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            log(f"     ✗ {last_error}")

        except requests.exceptions.Timeout:
            last_error = f"Attempt {attempt} timed out after 300s"
            log(f"     ✗ {last_error}")
        except Exception as e:
            last_error = str(e)
            log(f"     ✗ {last_error}")

        if attempt < 3:
            log(f"     Waiting {wait}s before retry…")
            time.sleep(wait)

    # ── All Supabase attempts failed → base64 fallback ────────────
    log(f"  ⚠  All upload attempts failed. Falling back to base64.")
    log(f"     Last error: {last_error}")
    log(f"     Encoding {mb:.1f} MB video as base64…")

    b64 = video_to_base64(video_path)
    b64_mb = len(b64) / 1_000_000
    log(f"     ✓ Base64 encoded ({b64_mb:.1f} MB string)")

    return {
        "video_url":      "",
        "video_base64":   b64,                      # decode this client-side to get the .mp4
        "video_filename": filename,
        "upload_method":  "base64_fallback",
        "upload_error":   last_error,
        "note":           "Supabase upload failed. Decode video_base64 to recover the mp4 file.",
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
#  Main handler
# ─────────────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    apply_hotfixes()
    os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
    os.environ["TORCH_ATTN_IMPLEMENTATION"]        = "eager"

    job_id = job["id"]
    inp    = job["input"]

    log(f"\n{'═' * 60}")
    log(f"  Job  : {job_id}")
    log(f"  Keys : {list(inp.keys())}")
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

        # ── 2. Audio mode ─────────────────────────────────────────
        audio_mode = inp.get("audio_mode", "audio")

        # ── 3. Build input JSON ───────────────────────────────────
        gen_input: dict = {
            "prompt":          inp.get("prompt", "A person talking naturally and expressively"),
            "negative_prompt": inp.get("negative_prompt", "distorted face, blurry, low quality"),
            "cond_image":      image_path,
        }

        if audio_mode == "audio":
            raw = inp.get("audio_urls") or inp.get("audio_url")
            if not raw:
                return {"error": "'audio_url' or 'audio_urls' required", "job_id": job_id}
            urls = [raw] if isinstance(raw, str) else list(raw)

            audio_paths: list[str] = []
            for i, url in enumerate(urls[:2]):
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
                gen_input["cond_audio"] = {"person1": audio_paths[0], "person2": audio_paths[1]}
                gen_input["audio_type"] = inp.get("audio_type", "para")

        elif audio_mode == "tts":
            tts_texts  = inp.get("tts_texts", [])
            tts_voices = inp.get("tts_voices", [])
            if not tts_texts:
                return {"error": "'tts_texts' required when audio_mode='tts'", "job_id": job_id}
            if isinstance(tts_texts, str):
                tts_texts = [tts_texts]

            voice_defaults = ["af_heart", "am_adam"]
            resolved = [
                voice_to_path(tts_voices[i] if i < len(tts_voices) else voice_defaults[min(i, 1)])
                for i in range(len(tts_texts[:2]))
            ]

            if len(tts_texts) == 1:
                gen_input["tts_audio"] = {"text": tts_texts[0], "human1_voice": resolved[0]}
                gen_input["cond_audio"] = {"person1": ""}
            else:
                gen_input["tts_audio"] = {
                    "text1": tts_texts[0], "text2": tts_texts[1],
                    "human1_voice": resolved[0], "human2_voice": resolved[1],
                }
                gen_input["cond_audio"] = {"person1": "", "person2": ""}
                gen_input["audio_type"] = inp.get("audio_type", "para")
        else:
            return {"error": f"Unknown audio_mode '{audio_mode}'", "job_id": job_id}

        input_json = str(tmp / "input.json")
        with open(input_json, "w") as fh:
            json.dump(gen_input, fh, indent=2)
        log(f"  input.json:\n{json.dumps(gen_input, indent=4)}\n")

        # ── 4. Generation parameters ──────────────────────────────
        save_stem      = str(Path(OUTPUT_DIR) / job_id)
        audio_save_dir = str(tmp / "save_audio")
        os.makedirs(audio_save_dir, exist_ok=True)

        mode            = inp.get("mode",           "clip")
        sample_steps    = int(inp.get("sample_steps",   20))
        size            = inp.get("size",           "multitalk-480")
        use_teacache    = bool(inp.get("use_teacache",  True))
        use_apg         = bool(inp.get("use_apg",       False))
        num_persistent  = int(inp.get("num_persistent_param_in_dit", 0))
        txt_guide       = float(inp.get("sample_text_guide_scale",  5.0))
        aud_guide       = float(inp.get("sample_audio_guide_scale", 4.0))
        teacache_thresh = float(inp.get("teacache_thresh", 0.3))
        sample_shift    = inp.get("sample_shift")
        lora_dir        = inp.get("lora_dir") or os.environ.get("LORA_DIR", "")
        lora_scale      = float(inp.get("lora_scale", 1.0))
        quant           = inp.get("quant",         "int8")
        quant_dir       = inp.get("quant_dir")     or MULTITALK_DIR
        offload_model   = inp.get("offload_model", False)

        log(f"  mode          : {mode}")
        log(f"  sample_steps  : {sample_steps}")
        log(f"  offload_model : {offload_model}")
        log(f"  quant         : {quant or 'disabled'}")

        # ── 5. Build command ──────────────────────────────────────
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
            "--offload_model",                 "False" if not offload_model else "True",
            "--save_file",                     save_stem,
        ]

        if use_teacache:
            cmd.append("--use_teacache")
        if use_apg:
            cmd.append("--use_apg")
        if audio_mode == "tts":
            cmd += ["--audio_mode", "tts"]
        if quant:
            cmd += ["--quant", quant, "--quant_dir", quant_dir]
        if lora_dir:
            cmd += ["--lora_dir", lora_dir, "--lora_scale", str(lora_scale)]
        if sample_shift is not None:
            cmd += ["--sample_shift", str(sample_shift)]

        log(f"\n  Command:\n  {' '.join(cmd)}\n")

        # ── 6. Run generation ─────────────────────────────────────
        log("  🚀 Launching generation…\n")
        subprocess_env = {**os.environ, "PYTHONUNBUFFERED": "1",
                          "TRANSFORMERS_ATTN_IMPLEMENTATION": "eager"}
        try:
            proc = subprocess.Popen(
                cmd, cwd="/app", env=subprocess_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )

            start      = time.time()
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
        video_path = find_output_video(save_stem)
        if not video_path:
            return {
                "error":               "Output video not found after generation",
                "save_stem":           save_stem,
                "output_dir_contents": [str(p) for p in Path(OUTPUT_DIR).iterdir()],
                "job_id":              job_id,
            }

        log(f"  ✓ Output: {video_path}")

        # ── 8. Upload (with base64 fallback) ──────────────────────
        upload_result = upload_video(video_path, job_id)

        # Clean up only when upload succeeded (keep file for debug on failure)
        if upload_result.get("upload_method") == "supabase":
            for p in Path(OUTPUT_DIR).glob(f"{job_id}*"):
                try:
                    p.unlink()
                except Exception:
                    pass

        # Always return success — video is either in video_url or video_base64
        status = "success" if upload_result.get("video_url") else "success_base64_fallback"
        return {"status": status, "job_id": job_id, **upload_result}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
