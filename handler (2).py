"""
RunPod Serverless Handler – MultiTalk (alive_and_speak)

Input schema:
{
    "image_url":    "https://…/person.jpg",          # required
    "audio_urls":   ["https://…/speech.wav"],         # required (audio_mode=audio)
    "prompt":       "A person talking naturally",      # optional
    "negative_prompt": "blurry, distorted face",       # optional
    "audio_mode":   "audio" | "tts",                  # default: audio
    "tts_texts":    ["Hello world"],                   # required when audio_mode=tts
    "audio_type":   "para",                           # multi-person sync mode
    "mode":         "streaming",                      # default: streaming
    "sample_steps": 40,                               # default: 40
    "size":         "multitalk-480",                  # default: multitalk-480
    "use_teacache": true,                             # default: true
    "sample_text_guide_scale":  5.0,
    "sample_audio_guide_scale": 4.0,
    "teacache_thresh":          0.3,
    "num_persistent_param_in_dit": 0,
    "sample_shift": null,
    "lora_dir":     "",
    "lora_scale":   1.0,
    "quant":        "",
    "quant_dir":    ""
}
"""

import os
import sys
import json
import requests
import tempfile
import subprocess
from pathlib import Path

import runpod

# ─────────────────────────────────────────────────────────────────
#  Environment / model paths
# ─────────────────────────────────────────────────────────────────
CKPT_DIR = os.environ.get(
    "CKPT_DIR", "/runpod-volume/models/Wan2.1-I2V-14B-480P"
)
WAV2VEC_DIR = os.environ.get(
    "WAV2VEC_DIR", "/runpod-volume/models/chinese-wav2vec2-base"
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/multitalk_outputs")
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "900"))  # 15 min

SIGNED_UPLOAD_ENDPOINT: str = os.environ.get(
    "SIGNED_UPLOAD_ENDPOINT",
    "https://kabdqrzcewkzbjmeqmxx.supabase.co/functions/v1/runpod-signed-upload",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(msg, flush=True)


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
    log(f"  ⬆  Uploading {video_path} …")
    mb = os.path.getsize(video_path) / 1_000_000
    log(f"     Size: {mb:.1f} MB")

    with open(video_path, "rb") as fh:
        resp = requests.post(
            SIGNED_UPLOAD_ENDPOINT,
            headers={
                "Content-Type": "video/mp4",
                "X-Job-Id":     job_id,
                "X-Filename":   f"{job_id}.mp4",
            },
            data=fh,
            timeout=300,
        )
    resp.raise_for_status()
    payload = resp.json()

    video_url = (
        payload.get("url")
        or payload.get("publicUrl")
        or payload.get("signedUrl")
        or ""
    )
    log(f"     ✓ {video_url}")
    return {"video_url": video_url, "upload_response": payload}


def find_output_video(save_stem: str) -> str | None:
    """
    generate_multitalk.py saves as {save_stem}.mp4.
    In streaming mode it may write chunks then concatenate in-place.
    Try a few candidate paths.
    """
    for suffix in ["", "_out", "_final", "_0"]:
        p = f"{save_stem}{suffix}.mp4"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p

    # Glob fallback in OUTPUT_DIR
    stem_name = Path(save_stem).name
    for p in sorted(Path(OUTPUT_DIR).glob(f"{stem_name}*.mp4")):
        if p.stat().st_size > 0:
            return str(p)

    return None


# ─────────────────────────────────────────────────────────────────
#  Handler
# ─────────────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    job_id = job["id"]
    inp    = job["input"]

    log(f"\n{'═' * 60}")
    log(f"  Job  : {job_id}")
    log(f"  Keys : {list(inp.keys())}")
    log(f"{'═' * 60}\n")

    with tempfile.TemporaryDirectory(prefix=f"mt_{job_id}_") as _tmp:
        tmp = Path(_tmp)

        # ── 1. Reference image ────────────────────────────────────
        image_url = inp.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required", "job_id": job_id}

        img_ext    = Path(image_url.split("?")[0]).suffix or ".jpg"
        image_path = str(tmp / f"ref_image{img_ext}")
        try:
            download_file(image_url, image_path)
        except Exception as exc:
            return {"error": f"Image download failed: {exc}", "job_id": job_id}

        # ── 2. Audio ──────────────────────────────────────────────
        audio_mode  = inp.get("audio_mode", "audio")   # "audio" | "tts"
        audio_paths: list[str] = []

        if audio_mode == "audio":
            audio_urls = inp.get("audio_urls") or inp.get("audio_url")
            if not audio_urls:
                return {
                    "error": "'audio_urls' is required when audio_mode='audio'",
                    "job_id": job_id,
                }
            if isinstance(audio_urls, str):
                audio_urls = [audio_urls]

            for i, url in enumerate(audio_urls[:2]):   # max 2 speakers
                a_ext  = Path(url.split("?")[0]).suffix or ".wav"
                a_path = str(tmp / f"audio_{i}{a_ext}")
                try:
                    download_file(url, a_path)
                    audio_paths.append(a_path)
                except Exception as exc:
                    return {
                        "error": f"Audio[{i}] download failed: {exc}",
                        "job_id": job_id,
                    }

        # ── 3. Build input JSON for generate_multitalk.py ─────────
        prompt     = inp.get("prompt", "A person talking naturally and expressively")
        neg_prompt = inp.get("negative_prompt", "distorted face, blurry, low quality")

        gen_input: dict = {
            "prompt":           prompt,
            "negative_prompt":  neg_prompt,
            "ref_image_path":   image_path,
        }

        if audio_mode == "audio":
            if len(audio_paths) == 1:
                gen_input["audio_path"] = audio_paths[0]
            elif len(audio_paths) == 2:
                gen_input["audio_path"]   = audio_paths[0]
                gen_input["audio_path_2"] = audio_paths[1]
                gen_input["audio_type"]   = inp.get("audio_type", "para")
        elif audio_mode == "tts":
            tts_texts = inp.get("tts_texts", [])
            if not tts_texts:
                return {
                    "error": "'tts_texts' is required when audio_mode='tts'",
                    "job_id": job_id,
                }
            if isinstance(tts_texts, str):
                tts_texts = [tts_texts]
            gen_input["audio_path"] = tts_texts[0]
            if len(tts_texts) > 1:
                gen_input["audio_path_2"] = tts_texts[1]
        else:
            return {"error": f"Unknown audio_mode: {audio_mode}", "job_id": job_id}

        input_json = str(tmp / "input.json")
        with open(input_json, "w") as fh:
            json.dump(gen_input, fh, indent=2)
        log(f"  input.json:\n{json.dumps(gen_input, indent=4)}\n")

        # ── 4. Generation parameters ──────────────────────────────
        save_stem       = str(Path(OUTPUT_DIR) / job_id)
        mode            = inp.get("mode", "streaming")
        sample_steps    = int(inp.get("sample_steps", 40))
        size            = inp.get("size", "multitalk-480")
        use_teacache    = bool(inp.get("use_teacache", True))
        num_persistent  = int(inp.get("num_persistent_param_in_dit", 0))
        txt_guide       = float(inp.get("sample_text_guide_scale", 5.0))
        aud_guide       = float(inp.get("sample_audio_guide_scale", 4.0))
        teacache_thresh = float(inp.get("teacache_thresh", 0.3))
        sample_shift    = inp.get("sample_shift")
        lora_dir        = inp.get("lora_dir") or os.environ.get("LORA_DIR", "")
        lora_scale      = float(inp.get("lora_scale", 1.0))
        quant           = inp.get("quant", "")
        quant_dir       = inp.get("quant_dir") or os.environ.get(
            "QUANT_DIR", "/runpod-volume/models/MeiGen-MultiTalk"
        )

        # ── 5. Assemble CLI command ───────────────────────────────
        cmd: list[str] = [
            sys.executable,                    "/app/generate_multitalk.py",
            "--ckpt_dir",                      CKPT_DIR,
            "--wav2vec_dir",                   WAV2VEC_DIR,
            "--input_json",                    input_json,
            "--sample_steps",                  str(sample_steps),
            "--mode",                          mode,
            "--size",                          size,
            "--num_persistent_param_in_dit",   str(num_persistent),
            "--sample_text_guide_scale",       str(txt_guide),
            "--sample_audio_guide_scale",      str(aud_guide),
            "--teacache_thresh",               str(teacache_thresh),
            "--save_file",                     save_stem,
        ]

        if use_teacache:
            cmd.append("--use_teacache")
        if audio_mode == "tts":
            cmd += ["--audio_mode", "tts"]
        if lora_dir:
            cmd += ["--lora_dir", lora_dir, "--lora_scale", str(lora_scale)]
        if quant:
            cmd += ["--quant", quant, "--quant_dir", quant_dir]
        if sample_shift is not None:
            cmd += ["--sample_shift", str(sample_shift)]

        log(f"  Command:\n  {' '.join(cmd)}\n")

        # ── 6. Run generation ─────────────────────────────────────
        try:
            proc = subprocess.run(
                cmd,
                cwd="/app",
                capture_output=True,
                text=True,
                timeout=GENERATION_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {
                "error":   f"Generation timed out after {GENERATION_TIMEOUT}s",
                "job_id":  job_id,
            }

        log(f"  Return code: {proc.returncode}")
        if proc.stdout:
            log(f"  STDOUT (tail):\n{proc.stdout[-3000:]}")
        if proc.stderr:
            log(f"  STDERR (tail):\n{proc.stderr[-3000:]}")

        if proc.returncode != 0:
            return {
                "error":      "Video generation failed",
                "returncode": proc.returncode,
                "stderr":     proc.stderr[-4000:],
                "stdout":     proc.stdout[-2000:],
                "job_id":     job_id,
            }

        # ── 7. Locate output video ────────────────────────────────
        video_path = find_output_video(save_stem)
        if not video_path:
            return {
                "error":               "Output video not found after generation",
                "save_stem":           save_stem,
                "output_dir_contents": [str(p) for p in Path(OUTPUT_DIR).iterdir()],
                "job_id":              job_id,
            }

        log(f"  ✓ Output found: {video_path}")

        # ── 8. Upload to Supabase signed endpoint ─────────────────
        try:
            upload_result = upload_video(video_path, job_id)
        except Exception as exc:
            return {"error": f"Upload failed: {exc}", "job_id": job_id}
        finally:
            # Clean up output file(s) for this job
            for p in Path(OUTPUT_DIR).glob(f"{job_id}*"):
                try:
                    p.unlink()
                except Exception:
                    pass

        return {
            "status":  "success",
            "job_id":  job_id,
            **upload_result,
        }


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
