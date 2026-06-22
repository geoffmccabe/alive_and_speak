"""
RunPod Serverless Handler – FLOAT + edge-tts
https://github.com/deepbrainai-research/float

Two input modes:
  A) Text → edge-tts → audio → FLOAT → video
  B) Audio URL → FLOAT → video

Top 20 English voices for "voice" parameter:
  US Male   : en-US-GuyNeural, en-US-ChristopherNeural, en-US-EricNeural,
               en-US-RogerNeural, en-US-SteffanNeural, en-US-AndrewNeural
  US Female : en-US-AriaNeural, en-US-JennyNeural, en-US-MichelleNeural,
               en-US-AvaNeural, en-US-EmmaNeural
  GB Male   : en-GB-RyanNeural, en-GB-ThomasNeural
  GB Female : en-GB-SoniaNeural, en-GB-LibbyNeural
  Other     : en-AU-WilliamMultilingualNeural, en-CA-LiamNeural,
               en-IN-PrabhatNeural, en-ZA-LukeNeural, en-IE-ConnorNeural

Input schema:
{
  "input": {
    "image_url":   "https://...",          # REQUIRED
    
    # ── Option A: Text-to-Speech ──────────────────────────────
    "text":        "Hello, how are you?",  # if set, edge-tts is used
    "voice":       "en-US-GuyNeural",      # default: en-US-GuyNeural
    
    # ── Option B: Direct audio ────────────────────────────────
    "audio_url":   "https://.../audio.wav",
    
    # ── FLOAT params ──────────────────────────────────────────
    "emotion":     null,                   # angry|disgust|fear|happy|neutral|sad|surprise
    "no_crop":     false,
    "a_cfg_scale": 2.0,
    "e_cfg_scale": 1.0,
    "r_cfg_scale": 1.0,
    "nfe":         10,
    "seed":        25
  }
}
"""

import os
import sys
import time
import base64
import asyncio
import subprocess
import tempfile
import requests
from pathlib import Path
from typing import Optional

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
CKPT_PATH  = os.environ.get("CKPT_PATH",  "/app/checkpoints/float.pth")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/float_outputs")
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "300"))

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
#  TTS
# ─────────────────────────────────────────────────────────────────
def text_to_wav(text: str, voice: str, wav_path: str) -> str:
    """Generate WAV from text using edge-tts. Returns wav_path."""
    import edge_tts

    if voice not in VOICES:
        raise ValueError(
            f"Unknown voice '{voice}'. Choose from: {list(VOICES.keys())}"
        )

    mp3_path = wav_path.replace(".wav", ".mp3")

    async def _generate():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(mp3_path)

    asyncio.run(_generate())

    # Convert MP3 → WAV 16kHz mono (required by wav2vec2 inside FLOAT)
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path,
         "-ar", "16000", "-ac", "1", wav_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(mp3_path)
    return wav_path


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


def upload_video(video_path: str, job_id: str) -> dict:
    mb       = os.path.getsize(video_path) / 1_000_000
    filename = f"{job_id}.mp4"
    log(f"  ⬆  Uploading {filename} ({mb:.1f} MB)")

    last_error = ""
    for attempt in range(1, 4):
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
                    timeout=300,
                )
            log(f"     HTTP {resp.status_code}")
            if resp.ok:
                payload   = resp.json()
                video_url = (payload.get("url") or payload.get("publicUrl")
                             or payload.get("signedUrl") or "")
                log(f"     ✓ {video_url}")
                return {"video_url": video_url, "upload_response": payload,
                        "upload_method": "supabase"}
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.exceptions.Timeout:
            last_error = f"Attempt {attempt} timed out"
        except Exception as e:
            last_error = str(e)
        log(f"     ✗ {last_error}")
        if attempt < 3:
            time.sleep(2 ** attempt)

    log("  ⚠  All uploads failed — encoding as base64 fallback")
    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "video_url":      "",
        "video_base64":   b64,
        "video_filename": filename,
        "upload_method":  "base64_fallback",
        "upload_error":   last_error,
        "note":           "Decode video_base64 to recover the mp4.",
    }


def find_output_video(path: str) -> Optional[str]:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    stem = Path(path).stem
    for p in sorted(Path(OUTPUT_DIR).glob(f"{stem}*.mp4")):
        if p.stat().st_size > 0:
            return str(p)
    return None


# ─────────────────────────────────────────────────────────────────
#  Main handler
# ─────────────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    job_id = job["id"]
    inp    = job["input"]

    log(f"\n{'═' * 60}")
    log(f"  Job  : {job_id}")
    log(f"  Keys : {list(inp.keys())}")
    log(f"{'═' * 60}\n")

    with tempfile.TemporaryDirectory(prefix=f"float_{job_id}_") as _tmp:
        tmp = Path(_tmp)

        # ── 1. Download reference image ───────────────────────────
        image_url = inp.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required", "job_id": job_id}

        img_ext    = Path(image_url.split("?")[0]).suffix or ".jpg"
        image_path = str(tmp / f"ref{img_ext}")
        try:
            download_file(image_url, image_path)
        except Exception as exc:
            return {"error": f"Image download failed: {exc}", "job_id": job_id}

        # ── 2. Prepare audio ──────────────────────────────────────
        audio_path = str(tmp / "audio.wav")
        text       = inp.get("text", "").strip()
        audio_url  = inp.get("audio_url") or inp.get("audio_urls")
        if isinstance(audio_url, list):
            audio_url = audio_url[0]

        if text:
            # Mode A: Text → edge-tts → WAV
            voice = inp.get("voice", DEFAULT_VOICE)
            log(f"  🎤 TTS mode")
            log(f"     Voice : {voice}  ({VOICES.get(voice, 'unknown')})")
            log(f"     Text  : {text[:80]}{'…' if len(text) > 80 else ''}")
            try:
                text_to_wav(text, voice, audio_path)
                mb = os.path.getsize(audio_path) / 1_000_000
                log(f"     ✓ Audio generated ({mb:.2f} MB)")
            except Exception as exc:
                return {"error": f"TTS failed: {exc}", "job_id": job_id}

        elif audio_url:
            # Mode B: Download audio file
            log(f"  🎵 Audio URL mode")
            aud_ext = Path(audio_url.split("?")[0]).suffix or ".wav"
            raw_audio = str(tmp / f"raw_audio{aud_ext}")
            try:
                download_file(audio_url, raw_audio)
            except Exception as exc:
                return {"error": f"Audio download failed: {exc}", "job_id": job_id}

            # Convert to 16kHz mono WAV if needed
            if aud_ext.lower() != ".wav":
                subprocess.run(
                    ["ffmpeg", "-y", "-i", raw_audio,
                     "-ar", "16000", "-ac", "1", audio_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                import shutil
                shutil.copy(raw_audio, audio_path)
        else:
            return {
                "error": "Provide either 'text' (TTS) or 'audio_url' (audio file)",
                "available_voices": VOICES,
                "job_id": job_id,
            }

        # ── 3. FLOAT parameters ───────────────────────────────────
        output_path = str(Path(OUTPUT_DIR) / f"{job_id}.mp4")
        emotion     = inp.get("emotion")       # None = S2E (auto from audio)
        no_crop     = bool(inp.get("no_crop",      False))
        a_cfg_scale = float(inp.get("a_cfg_scale",  2.0))
        e_cfg_scale = float(inp.get("e_cfg_scale",  1.0))
        r_cfg_scale = float(inp.get("r_cfg_scale",  1.0))
        nfe         = int(inp.get("nfe",            10))
        seed        = int(inp.get("seed",           25))

        log(f"\n  emotion     : {emotion or 'S2E (auto)'}")
        log(f"  nfe         : {nfe}  (flow steps)")
        log(f"  a_cfg_scale : {a_cfg_scale}")
        log(f"  e_cfg_scale : {e_cfg_scale}")

        # ── 4. Build CLI command ──────────────────────────────────
        cmd: list[str] = [
            sys.executable,     "/app/generate.py",
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

        log(f"\n  Command:\n  {' '.join(cmd)}\n")

        # ── 5. Run generation ─────────────────────────────────────
        log("  🚀 Launching FLOAT…\n")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd="/app",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
                    return {"error": f"Timed out after {GENERATION_TIMEOUT}s",
                            "job_id": job_id}

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

        # ── 6. Find output ────────────────────────────────────────
        video_path = find_output_video(output_path)
        if not video_path:
            return {
                "error":               "Output video not found",
                "expected_path":       output_path,
                "output_dir_contents": [str(p) for p in Path(OUTPUT_DIR).iterdir()],
                "job_id":              job_id,
            }

        log(f"  ✓ Output: {video_path}")

        # ── 7. Upload ─────────────────────────────────────────────
        upload_result = upload_video(video_path, job_id)

        if upload_result.get("upload_method") == "supabase":
            try:
                os.remove(video_path)
            except Exception:
                pass

        status = "success" if upload_result.get("video_url") else "success_base64_fallback"
        return {"status": status, "job_id": job_id, **upload_result}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
