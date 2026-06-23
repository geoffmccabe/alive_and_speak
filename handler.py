"""
RunPod Serverless Handler – FLOAT + edge-tts
Compatible with older Python runtim(no PEP 604 union syntax).

Two input modes:
  A) Text → edge-ts → audio → FLOAT → video
  B) Audio URL → FLOAT → video
"""

import os
import sys
import time
import base64
import asyncio
import subprocess
import tempfile
import traceback
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import runpod

# ─────────────────────────────────────────────────────────────────
#  Available voices reference
# ─────────────────────────────────────────────────────────────────
VOICES = {
    # US Male
    "en-US-GuyNeural":         "US Male – neutral, authoritative",
    "en-US-ChristopherNeural": "US Male – warm, conversational",
    "en-US-EricNeural":        "US Male – clear, professional",
    "en-US-RogerNeural":       "US Male – energetic",
    "en-US-SteffanNeural":     "US Male – deep, newscast",
    "en-US-AndrewNeural":      "US Male – friendly",
    # US Female
    "en-US-AriaNeural":        "US Female – lively, expressive",
    "en-US-JennyNeural":       "US Female – warm, assistant-style",
    "en-US-MichelleNeural":    "US Female – friendly, positive",
    "en-US-AvaNeural":         "US Female – gentle, soothing",
    "en-US-EmmaNeural":        "US Female – confident",
    # GB
    "en-GB-RyanNeural":        "British Male – crisp, formal",
    "en-GB-ThomasNeural":      "British Male – authoritative",
    "en-GB-SoniaNeural":       "British Female – natural, warm",
    "en-GB-LibbyNeural":       "British Female – clear",
    # Other accents
    "en-AU-WilliamMultilingualNeural": "Australian Male – relaxed",
    "en-CA-LiamNeural":        "Canadian Male – friendly",
    "en-IN-PrabhatNeural":     "Indian Male – distinct accent",
    "en-ZA-LukeNeural":        "South African Male – unique",
    "en-IE-ConnorNeural":      "Irish Male – warm brogue",
}

DEFAULT_VOICE = "en-US-GuyNeural"

# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────
CKPT_PATH = os.environ.get("CKPT_PATH", "/app/checkpoints/float.pth")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/float_outputs")
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "300"))

SIGNED_UPLOAD_ENDPOINT = os.environ.get(
    "SIGNED_UPLOAD_ENDPOINT",
    "https://kabdqrzcewkzbjmeqmxx.supabase.co/functions/v1/runpod-signed-upload",
)
RUNPOD_UPLOAD_SECRET = os.environ.get("RUNPOD_UPLOAD_SECRET", "")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  Logging / error helpers
# ─────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(msg, flush=True)


def error_response(
    job_id: str,
    stage: str,
    exc: Exception,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tb = traceback.format_exc()
    payload: Dict[str, Any] = {
        "error": str(exc),
        "stage": stage,
        "job_id": job_id,
        "traceback": tb[-4000:],
    }
    if extra:
        payload.update(extra)
    return payload


def run_coro_safely(coro):
    """
    Run an async coroutine safely from sync code.

    - If no loop is running, uses asyncio.run().
    - If a loop is already running, executes the coroutine in a new thread
      with its own event loop.
    """
    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        loop_running = False

    if not loop_running:
        return asyncio.run(coro)

    def _runner():
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_runner).result()


# ─────────────────────────────────────────────────────────────────
#  TTS
# ─────────────────────────────────────────────────────────────────
def text_to_wav(text: str, voice: str, wav_path: str) -> str:
    """Generate WAV from text using edge-tts. Returns wav_path."""
    import edge_tts

    if voice not in VOICES:
        raise ValueError("Unknown voice '%s'. Choose from: %s" % (voice, list(VOICES.keys())))

    mp3_path = wav_path.replace(".wav", ".mp3")

    async def _generate():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(mp3_path)

    run_coro_safely(_generate())

    # Convert MP3 → WAV 16kHz mono (required by wav2vec2 inside FLOAT)
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(mp3_path)
    return wav_path


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────
def download_file(url: str, dest: str, timeout: int = 120) -> str:
    log("  ⬇  %s" % url)
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=65_536):
            fh.write(chunk)
    mb = os.path.getsize(dest) / 1_000_000
    log("     ✓ %.1f MB" % mb)
    return dest


def upload_video(video_path: str, job_id: str) -> Dict[str, Any]:
    mb = os.path.getsize(video_path) / 1_000_000
    filename = "%s.mp4" % job_id
    log("  ⬆  Uploading %s (%.1f MB)" % (filename, mb))

    if not RUNPOD_UPLOAD_SECRET:
        return {
            "video_url": "",
            "upload_method": "missing_secret",
            "upload_error": "RUNPOD_UPLOAD_SECRET is not set",
            "video_filename": filename,
        }

    last_error = ""
    for attempt in range(1, 4):
        log("     Attempt %d/3 …" % attempt)
        try:
            with open(video_path, "rb") as fh:
                resp = requests.post(
                    SIGNED_UPLOAD_ENDPOINT,
                    headers={
                        "Authorization": "Bearer %s" % RUNPOD_UPLOAD_SECRET,
                        "Content-Type": "video/mp4",
                        "X-Job-Id": job_id,
                        "X-Filename": filename,
                    },
                    data=fh,
                    timeout=300,
                )
            log("     HTTP %s" % resp.status_code)
            if resp.ok:
                payload = resp.json()
                video_url = (
                    payload.get("url")
                    or payload.get("publicUrl")
                    or payload.get("signedUrl")
                    or ""
                )
                log("     ✓ %s" % video_url)
                return {
                    "video_url": video_url,
                    "upload_response": payload,
                    "upload_method": "supabase",
                    "video_filename": filename,
                }
            last_error = "HTTP %s: %s" % (resp.status_code, resp.text[:200])
        except requests.exceptions.Timeout:
            last_error = "Attempt %d timed out" % attempt
        except Exception as e:
            last_error = str(e)

        log("     ✗ %s" % last_error)
        if attempt < 3:
            time.sleep(2 ** attempt)

    log("  ⚠  All uploads failed — encoding as base64 fallback")
    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "video_url": "",
        "video_base64": b64,
        "video_filename": filename,
        "upload_method": "base64_fallback",
        "upload_error": last_error,
        "note": "Decode video_base64 to recover the mp4.",
    }


def find_output_video(path: str) -> Optional[str]:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    stem = Path(path).stem
    for p in sorted(Path(OUTPUT_DIR).glob("%s*.mp4" % stem)):
        if p.stat().st_size > 0:
            return str(p)
    return None


# ─────────────────────────────────────────────────────────────────
#  Main handler
# ─────────────────────────────────────────────────────────────────
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_id = job.get("id", "unknown")
    inp = job.get("input", {})

    log("\n%s" % ("═" * 60))
    log("  Job  : %s" % job_id)
    log("  Keys : %s" % list(inp.keys()))
    log("%s\n" % ("═" * 60))

    try:
        with tempfile.TemporaryDirectory(prefix="float_%s_" % job_id) as _tmp:
            tmp = Path(_tmp)

            # ── 1. Download reference image ──────────────────────
            image_url = inp.get("image_url")
            if not image_url:
                return {"error": "'image_url' is required", "job_id": job_id, "stage": "validate_input"}

            img_ext = Path(image_url.split("?")[0]).suffix or ".jpg"
            image_path = str(tmp / ("ref" + img_ext))

            try:
                download_file(image_url, image_path)
            except Exception as exc:
                return error_response(job_id, "image_download", exc)

            # ── 2. Prepare audio ──────────────────────────────────
            audio_path = str(tmp / "audio.wav")
            text = (inp.get("text") or "").strip()
            audio_url = inp.get("audio_url") or inp.get("audio_urls")
            if isinstance(audio_url, list):
                audio_url = audio_url[0]

            if text:
                voice = inp.get("voice", DEFAULT_VOICE)
                log("  🎤 TTS mode")
                log("     Voice : %s  (%s)" % (voice, VOICES.get(voice, "unknown")))
                log("     Text  : %s%s" % (text[:80], "…" if len(text) > 80 else ""))

                try:
                    text_to_wav(text, voice, audio_path)
                    mb = os.path.getsize(audio_path) / 1_000_000
                    log("     ✓ Audio generated (%.2f MB)" % mb)
                except Exception as exc:
                    return error_response(job_id, "tts", exc, {"voice": voice})

            elif audio_url:
                log("  🎵 Audio URL mode")
                aud_ext = Path(audio_url.split("?")[0]).suffix or ".wav"
                raw_audio = str(tmp / ("raw_audio" + aud_ext))

                try:
                    download_file(audio_url, raw_audio)
                except Exception as exc:
                    return error_response(job_id, "audio_download", exc)

                try:
                    if aud_ext.lower() != ".wav":
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", raw_audio, "-ar", "16000", "-ac", "1", audio_path],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        import shutil
                        shutil.copy(raw_audio, audio_path)
                except Exception as exc:
                    return error_response(job_id, "audio_convert", exc)
            else:
                return {
                    "error": "Provide either 'text' (TTS) or 'audio_url' (audio file)",
                    "available_voices": VOICES,
                    "job_id": job_id,
                    "stage": "validate_input",
                }

            # ── 3. FLOAT parameters ───────────────────────────────
            output_path = str(Path(OUTPUT_DIR) / ("%s.mp4" % job_id))
            emotion = inp.get("emotion")
            no_crop = bool(inp.get("no_crop", False))
            a_cfg_scale = float(inp.get("a_cfg_scale", 2.0))
            e_cfg_scale = float(inp.get("e_cfg_scale", 1.0))
            r_cfg_scale = float(inp.get("r_cfg_scale", 1.0))
            nfe = int(inp.get("nfe", 10))
            seed = int(inp.get("seed", 25))

            log("\n  emotion     : %s" % (emotion or "S2E (auto)"))
            log("  nfe         : %s  (flow steps)" % nfe)
            log("  a_cfg_scale : %s" % a_cfg_scale)
            log("  e_cfg_scale : %s" % e_cfg_scale)

            # ── 4. Build CLI command ──────────────────────────────
            cmd = [
                sys.executable, "/app/generate.py",
                "--ref_path", image_path,
                "--aud_path", audio_path,
                "--res_video_path", output_path,
                "--ckpt_path", CKPT_PATH,
                "--a_cfg_scale", str(a_cfg_scale),
                "--e_cfg_scale", str(e_cfg_scale),
                "--r_cfg_scale", str(r_cfg_scale),
                "--nfe", str(nfe),
                "--seed", str(seed),
            ]
            if emotion:
                cmd += ["--emo", emotion]
            if no_crop:
                cmd.append("--no_crop")

            log("\n  Command:\n  %s\n" % " ".join(cmd))

            # ── 5. Run generation ────────────────────────────────
            log("  🚀 Launching FLOAT…\n")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd="/app",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=dict(os.environ, PYTHONUNBUFFERED="1"),
                )

                start = time.time()
                lines: List[str] = []

                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip("\r\n")
                    log(line)
                    lines.append(line)

                    if time.time() - start > GENERATION_TIMEOUT:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except Exception:
                            proc.kill()
                        return {
                            "error": "Timed out after %ds" % GENERATION_TIMEOUT,
                            "job_id": job_id,
                            "stage": "generation_timeout",
                        }

                return_code = proc.wait()

            except Exception as exc:
                return error_response(job_id, "generation_subprocess", exc)

            log("\n  Return code: %s" % return_code)

            if return_code != 0:
                return {
                    "error": "Video generation failed",
                    "returncode": return_code,
                    "output": "\n".join(lines[-80:]),
                    "job_id": job_id,
                    "stage": "generation_failed",
                }

            # ── 6. Find output ────────────────────────────────────
            video_path = find_output_video(output_path)
            if not video_path:
                return {
                    "error": "Output video not found",
                    "expected_path": output_path,
                    "output_dir_contents": [str(p) for p in Path(OUTPUT_DIR).iterdir()],
                    "job_id": job_id,
                    "stage": "output_missing",
                }

            log("  ✓ Output: %s" % video_path)

            # ── 7. Upload ─────────────────────────────────────────
            try:
                upload_result = upload_video(video_path, job_id)
            except Exception as exc:
                return error_response(job_id, "upload", exc)

            if upload_result.get("upload_method") == "supabase":
                try:
                    os.remove(video_path)
                except Exception:
                    pass

            status = "success" if upload_result.get("video_url") else "success_base64_fallback"
            return {"status": status, "job_id": job_id, **upload_result}

    except Exception as exc:
        return error_response(job_id, "top_level", exc)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
