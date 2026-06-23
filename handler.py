"""
RunPod Serverless Handler – FLOAT + edge-tts
Output: base64-en returned directly in the job response.
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
#  Voices
# ─────────────────────────────────────────────────────────────────
VOICES = {
    "en-US-GuyNeural":                  "US Male – neutral, authoritative",
    "en-US-ChristopherNeural":          "US Male – warm, conversational",
    "en-US-EricNeural":                 "US Male – clear, professional",
    "en-US-RogerNeural":                "US Male – energetic",
    "en-US-SteffanNeural":              "US Male – deep, newscast",
    "en-US-AndrewNeural":               "US Male – friendly",
    "en-US-AriaNeural":                 "US Female – lively, expressive",
    "en-US-JennyNeural":                "US Female – warm, assistant-style",
    "en-US-MichelleNeural":             "US Female – friendly, positive",
    "en-US-AvaNeural":                  "US Female – gentle, soothing",
    "en-US-EmmaNeural":                 "US Female – confident",
    "en-GB-RyanNeural":                 "British Male – crisp, formal",
    "en-GB-ThomasNeural":               "British Male – authoritative",
    "en-GB-SoniaNeural":                "British Female – natural, warm",
    "en-GB-LibbyNeural":                "British Female – clear",
    "en-AU-WilliamMultilingualNeural":  "Australian Male – relaxed",
    "en-CA-LiamNeural":                 "Canadian Male – friendly",
    "en-IN-PrabhatNeural":              "Indian Male – distinct accent",
    "en-ZA-LukeNeural":                 "South African Male – unique",
    "en-IE-ConnorNeural":               "Irish Male – warm brogue",
}
DEFAULT_VOICE = "en-US-GuyNeural"

# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────
CKPT_PATH          = os.environ.get("CKPT_PATH",          "/app/checkpoints/float.pth")
OUTPUT_DIR         = os.environ.get("OUTPUT_DIR",         "/tmp/float_outputs")
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "600"))  # 10 min for CPU

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(msg, flush=True)


def error_response(job_id, stage, exc, extra=None):
    payload = {
        "error":     str(exc),
        "stage":     stage,
        "job_id":    job_id,
        "traceback": traceback.format_exc()[-3000:],
    }
    if extra:
        payload.update(extra)
    return payload


def run_coro_safely(coro):
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


def text_to_wav(text: str, voice: str, wav_path: str) -> str:
    import edge_tts
    if voice not in VOICES:
        raise ValueError("Unknown voice '%s'. Choose from: %s" % (voice, list(VOICES.keys())))
    mp3_path = wav_path.replace(".wav", ".mp3")

    async def _generate():
        await edge_tts.Communicate(text, voice).save(mp3_path)

    run_coro_safely(_generate())
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.remove(mp3_path)
    return wav_path


def download_file(url: str, dest: str, timeout: int = 120) -> str:
    log("  ⬇  %s" % url)
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=65_536):
            fh.write(chunk)
    log("     ✓ %.1f MB" % (os.path.getsize(dest) / 1_000_000))
    return dest


def video_to_base64(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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
    inp    = job.get("input", {})

    log("\n%s" % ("═" * 60))
    log("  Job  : %s" % job_id)
    log("  Keys : %s" % list(inp.keys()))
    log("%s\n" % ("═" * 60))

    try:
        with tempfile.TemporaryDirectory(prefix="float_%s_" % job_id) as _tmp:
            tmp = Path(_tmp)

            # ── 1. Reference image ────────────────────────────────
            image_url = inp.get("image_url")
            if not image_url:
                return {"error": "'image_url' is required", "job_id": job_id}

            img_ext    = Path(image_url.split("?")[0]).suffix or ".jpg"
            image_path = str(tmp / ("ref" + img_ext))
            try:
                download_file(image_url, image_path)
            except Exception as exc:
                return error_response(job_id, "image_download", exc)

            # ── 2. Audio ──────────────────────────────────────────
            audio_path = str(tmp / "audio.wav")
            text       = (inp.get("text") or "").strip()
            audio_url  = inp.get("audio_url") or inp.get("audio_urls")
            if isinstance(audio_url, list):
                audio_url = audio_url[0]

            if text:
                voice = inp.get("voice", DEFAULT_VOICE)
                log("  🎤 TTS  voice=%s" % voice)
                log("     Text : %s%s" % (text[:80], "…" if len(text) > 80 else ""))
                try:
                    text_to_wav(text, voice, audio_path)
                    log("     ✓ %.2f MB" % (os.path.getsize(audio_path) / 1_000_000))
                except Exception as exc:
                    return error_response(job_id, "tts", exc, {"voice": voice})

            elif audio_url:
                log("  🎵 Audio URL")
                aud_ext   = Path(audio_url.split("?")[0]).suffix or ".wav"
                raw_audio = str(tmp / ("raw" + aud_ext))
                try:
                    download_file(audio_url, raw_audio)
                except Exception as exc:
                    return error_response(job_id, "audio_download", exc)
                try:
                    if aud_ext.lower() != ".wav":
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", raw_audio,
                             "-ar", "16000", "-ac", "1", audio_path],
                            check=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                    else:
                        import shutil
                        shutil.copy(raw_audio, audio_path)
                except Exception as exc:
                    return error_response(job_id, "audio_convert", exc)
            else:
                return {
                    "error": "Provide 'text' (TTS) or 'audio_url'",
                    "available_voices": VOICES,
                    "job_id": job_id,
                }

            # ── 3. FLOAT params ───────────────────────────────────
            output_path = str(Path(OUTPUT_DIR) / ("%s.mp4" % job_id))
            emotion     = inp.get("emotion")
            no_crop     = bool(inp.get("no_crop",     False))
            a_cfg_scale = float(inp.get("a_cfg_scale", 2.0))
            e_cfg_scale = float(inp.get("e_cfg_scale", 1.0))
            r_cfg_scale = float(inp.get("r_cfg_scale", 1.0))
            nfe         = int(inp.get("nfe",           10))
            seed        = int(inp.get("seed",          25))

            log("  emotion=%s  nfe=%s  a_cfg=%.1f  e_cfg=%.1f" % (
                emotion or "S2E", nfe, a_cfg_scale, e_cfg_scale))

            # ── 4. CLI command ────────────────────────────────────
            cmd = [
                sys.executable, "/app/generate.py",
                "--ref_path",       image_path,
                "--aud_path",       audio_path,
                "--res_video_path", output_path,
                "--ckpt_path",      CKPT_PATH,
                "--a_cfg_scale",    str(a_cfg_scale),
                "--e_cfg_scale",    str(e_cfg_scale),
                "--r_cfg_scale",    str(r_cfg_scale),
                "--nfe",            str(nfe),
                "--seed",           str(seed),
            ]
            if emotion:
                cmd += ["--emo", emotion]
            if no_crop:
                cmd.append("--no_crop")

            log("  CMD: %s\n" % " ".join(cmd))

            # ── 5. Run generation ────────────────────────────────
            log("  🚀 Running FLOAT…\n")
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
                for line in proc.stdout:
                    line = line.rstrip("\r\n")
                    log(line)
                    lines.append(line)
                    if time.time() - start > GENERATION_TIMEOUT:
                        proc.terminate()
                        try: proc.wait(timeout=10)
                        except Exception: proc.kill()
                        return {"error": "Timed out after %ds" % GENERATION_TIMEOUT,
                                "job_id": job_id}
                return_code = proc.wait()
            except Exception as exc:
                return error_response(job_id, "subprocess", exc)

            log("\n  Return code: %d" % return_code)

            if return_code != 0:
                return {
                    "error":      "Generation failed",
                    "returncode": return_code,
                    "output":     "\n".join(lines[-80:]),
                    "job_id":     job_id,
                }

            # ── 6. Find output ────────────────────────────────────
            video_path = find_output_video(output_path)
            if not video_path:
                return {
                    "error":               "Output video not found",
                    "expected_path":       output_path,
                    "output_dir_contents": [str(p) for p in Path(OUTPUT_DIR).iterdir()],
                    "job_id":              job_id,
                }

            mb = os.path.getsize(video_path) / 1_000_000
            log("  ✓ Output: %s (%.1f MB)" % (video_path, mb))

            # ── 7. Encode as base64 and return ────────────────────
            log("  📦 Encoding video as base64…")
            b64 = video_to_base64(video_path)
            log("  ✓ Done — %.1f MB base64 string" % (len(b64) / 1_000_000))

            # Clean up
            try:
                os.remove(video_path)
            except Exception:
                pass

            return {
                "status":         "success",
                "job_id":         job_id,
                "video_base64":   b64,
                "video_filename": "%s.mp4" % job_id,
                "video_size_mb":  round(mb, 2),
                "note":           "Decode video_base64 with base64.b64decode() to get the .mp4 bytes.",
            }

    except Exception as exc:
        return error_response(job_id, "top_level", exc)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
