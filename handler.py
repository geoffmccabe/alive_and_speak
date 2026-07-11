

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

import numpy as np
import requests
import runpod

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
VOICES = {
    "en-US-GuyNeural":                 "US Male – neutral, authoritative",
    "en-US-ChristopherNeural":         "US Male – warm, conversational",
    "en-US-EricNeural":                "US Male – clear, professional",
    "en-US-RogerNeural":               "US Male – energetic",
    "en-US-SteffanNeural":             "US Male – deep, newscast",
    "en-US-AndrewNeural":              "US Male – friendly",
    "en-US-AriaNeural":                "US Female – lively, expressive",
    "en-US-JennyNeural":               "US Female – warm, assistant-style",
    "en-US-MichelleNeural":            "US Female – friendly, positive",
    "en-US-AvaNeural":                 "US Female – gentle, soothing",
    "en-US-EmmaNeural":                "US Female – confident",
    "en-GB-RyanNeural":                "British Male – crisp, formal",
    "en-GB-ThomasNeural":              "British Male – authoritative",
    "en-GB-SoniaNeural":               "British Female – natural, warm",
    "en-GB-LibbyNeural":               "British Female – clear",
    "en-AU-WilliamMultilingualNeural": "Australian Male – relaxed",
    "en-CA-LiamNeural":                "Canadian Male – friendly",
    "en-IN-PrabhatNeural":             "Indian Male – distinct accent",
    "en-ZA-LukeNeural":                "South African Male – unique",
    "en-IE-ConnorNeural":              "Irish Male – warm brogue",
}
DEFAULT_VOICE = "en-US-GuyNeural"

# ─────────────────────────────────────────────────────────────────
#  Voice Effect Presets
# ─────────────────────────────────────────────────────────────────
VOICE_EFFECT_PRESETS = {
    "none":  dict(pitch_shift_semitones=0,   speed_factor=1.0,  growl_layers=0, distortion_amount=0.0),
    "ogre":  dict(pitch_shift_semitones=-5,  speed_factor=0.95, growl_layers=1, distortion_amount=0.10),
    "troll": dict(pitch_shift_semitones=-7,  speed_factor=0.92, growl_layers=2, distortion_amount=0.15),
    "demon": dict(pitch_shift_semitones=-10, speed_factor=0.88, growl_layers=3, distortion_amount=0.25),
    "abyss": dict(pitch_shift_semitones=-12, speed_factor=0.85, growl_layers=3, distortion_amount=0.35),
    "robot": None,  # handled separately
}

# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────
CKPT_PATH          = os.environ.get("CKPT_PATH",          "/app/checkpoints/float.pth")
OUTPUT_DIR         = os.environ.get("OUTPUT_DIR",         "/tmp/float_outputs")
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "600"))

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
#  TTS
# ─────────────────────────────────────────────────────────────────
def text_to_wav(text: str, voice: str, wav_path: str) -> str:
    import edge_tts
    if voice not in VOICES:
        raise ValueError("Unknown voice '%s'." % voice)
    mp3_path = wav_path.replace(".wav", ".mp3")

    async def _generate():
        await edge_tts.Communicate(text, voice).save(mp3_path)

    run_coro_safely(_generate())
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "22050", "-ac", "1", wav_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.remove(mp3_path)
    return wav_path


# ─────────────────────────────────────────────────────────────────
#  Voice Effects
# ─────────────────────────────────────────────────────────────────
def apply_monster_effect(
    wav_in: str,
    wav_out: str,
    pitch_shift_semitones: int   = -7,
    speed_factor: float          = 0.92,
    growl_layers: int            = 2,
    distortion_amount: float     = 0.15,
    delay_ms: int                = 90,
    decay_db: int                = 6,
    reverb_repeats: int          = 3,
) -> str:
    import librosa
    import soundfile as sf
    from pydub import AudioSegment
    from pydub.effects import normalize

    log("     Applying monster effect (pitch=%d, speed=%.2f, layers=%d, dist=%.2f)" % (
        pitch_shift_semitones, speed_factor, growl_layers, distortion_amount))

    y, sr = librosa.load(wav_in, sr=None)

    # 1. Slow down
    if speed_factor != 1.0:
        y = librosa.effects.time_stretch(y, rate=speed_factor)

    # 2. Pitch shift down
    if pitch_shift_semitones != 0:
        y_deep = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift_semitones)
    else:
        y_deep = y.copy()

    # 3. Growl layers (detuned copies)
    mix = y_deep.copy()
    for i in range(growl_layers):
        detune = pitch_shift_semitones - (i + 1) * 2
        layer  = librosa.effects.pitch_shift(y, sr=sr, n_steps=detune)
        layer  = layer * 0.5
        n      = min(len(mix), len(layer))
        mix    = mix[:n] + layer[:n]

    # 4. Tanh distortion / saturation
    if distortion_amount > 0:
        mix = np.tanh(mix * (1 + distortion_amount * 5))

    # 5. Normalize
    mix = mix / (np.max(np.abs(mix)) + 1e-6)

    # Write intermediate
    tmp_monster = wav_in.replace(".wav", "_monster_tmp.wav")
    sf.write(tmp_monster, mix, sr)

    # 6. Echo / reverb via pydub
    audio    = AudioSegment.from_wav(tmp_monster)
    combined = audio
    delayed  = audio
    for _ in range(reverb_repeats):
        delayed  = delayed - decay_db
        delayed  = AudioSegment.silent(duration=delay_ms) + delayed
        combined = combined.overlay(delayed)
    combined = normalize(combined)
    combined.export(wav_out, format="wav")

    os.remove(tmp_monster)
    log("     ✓ Monster effect applied → %s" % wav_out)
    return wav_out


def apply_robot_effect(wav_in: str, wav_out: str) -> str:
    """
    Robot voice via ffmpeg:
      - ring modulation at 50 Hz (creates metallic carrier tone)
      - slight pitch up (+2 semitones) for tinny feel
      - echo for metallic resonance
    """
    log("     Applying robot effect")
    cmd = [
        "ffmpeg", "-y", "-i", wav_in,
        "-af",
        (
            "asetrate=22050*1.05,"          # slight pitch up
            "aresample=22050,"
            "tremolo=f=50:d=0.9,"           # ring-mod style tremolo at 50 Hz
            "aecho=0.8:0.8:20|40:0.5|0.3," # metallic echo
            "equalizer=f=300:t=o:w=200:g=-6,"  # cut muddy mids
            "equalizer=f=3000:t=o:w=500:g=4"   # boost presence/metallic highs
        ),
        wav_out
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log("     ✓ Robot effect applied → %s" % wav_out)
    return wav_out


def apply_voice_effect(wav_in: str, effect: str, params: dict) -> str:
    """Apply the requested effect. Returns path to processed WAV."""
    if effect == "none":
        return wav_in

    wav_out = wav_in.replace(".wav", "_fx.wav")

    if effect == "robot":
        return apply_robot_effect(wav_in, wav_out)

    # Monster presets — get base params from preset then override with user params
    preset = VOICE_EFFECT_PRESETS.get(effect, VOICE_EFFECT_PRESETS["troll"]).copy()
    preset.update({k: v for k, v in params.items() if v is not None})
    return apply_monster_effect(wav_in, wav_out, **preset)


# ─────────────────────────────────────────────────────────────────
#  Final WAV → 16kHz mono for FLOAT
# ─────────────────────────────────────────────────────────────────
def resample_for_float(wav_in: str, wav_out: str) -> str:
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_in, "-ar", "16000", "-ac", "1", wav_out],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return wav_out


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

            # ── 2. Voice effect settings ──────────────────────────
            voice_effect = inp.get("voice_effect", "none").lower().strip()
            if voice_effect not in VOICE_EFFECT_PRESETS:
                return {
                    "error": "Unknown voice_effect '%s'. Choose from: %s" % (
                        voice_effect, list(VOICE_EFFECT_PRESETS.keys())),
                    "job_id": job_id,
                }

            # User can override individual effect params
            effect_params = {
                "pitch_shift_semitones": inp.get("pitch_shift_semitones"),
                "speed_factor":          inp.get("speed_factor"),
                "growl_layers":          inp.get("growl_layers"),
                "distortion_amount":     inp.get("distortion_amount"),
            }

            # ── 3. Get raw audio (TTS or URL) ─────────────────────
            raw_wav = str(tmp / "raw.wav")
            text    = (inp.get("text") or "").strip()
            audio_url = inp.get("audio_url") or inp.get("audio_urls")
            if isinstance(audio_url, list):
                audio_url = audio_url[0]

            if text:
                voice = inp.get("voice", DEFAULT_VOICE)
                log("  🎤 TTS  voice=%s  effect=%s" % (voice, voice_effect))
                log("     Text : %s%s" % (text[:80], "…" if len(text) > 80 else ""))
                try:
                    text_to_wav(text, voice, raw_wav)
                except Exception as exc:
                    return error_response(job_id, "tts", exc, {"voice": voice})

            elif audio_url:
                log("  🎵 Audio URL  effect=%s" % voice_effect)
                aud_ext   = Path(audio_url.split("?")[0]).suffix or ".wav"
                raw_dl    = str(tmp / ("raw_dl" + aud_ext))
                try:
                    download_file(audio_url, raw_dl)
                except Exception as exc:
                    return error_response(job_id, "audio_download", exc)
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", raw_dl, "-ar", "22050", "-ac", "1", raw_wav],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                except Exception as exc:
                    return error_response(job_id, "audio_convert", exc)
            else:
                return {
                    "error": "Provide 'text' (TTS) or 'audio_url'",
                    "available_voices": VOICES,
                    "available_effects": list(VOICE_EFFECT_PRESETS.keys()),
                    "job_id": job_id,
                }

            # ── 4. Apply voice effect ─────────────────────────────
            try:
                effected_wav = apply_voice_effect(raw_wav, voice_effect, effect_params)
            except Exception as exc:
                return error_response(job_id, "voice_effect", exc)

            # ── 5. Resample to 16kHz mono for FLOAT ──────────────
            final_wav = str(tmp / "audio.wav")
            try:
                resample_for_float(effected_wav, final_wav)
                log("     ✓ Audio ready (%.2f MB)" % (os.path.getsize(final_wav) / 1_000_000))
            except Exception as exc:
                return error_response(job_id, "resample", exc)

            # ── 6. FLOAT params ───────────────────────────────────
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

            # ── 7. Build CLI command ──────────────────────────────
            cmd = [
                sys.executable, "/app/generate.py",
                "--ref_path",       image_path,
                "--aud_path",       final_wav,
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

            # ── 8. Run FLOAT ──────────────────────────────────────
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

            # ── 9. Find output ────────────────────────────────────
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

            # ── 10. Return base64 ─────────────────────────────────
            log("  📦 Encoding as base64…")
            b64 = video_to_base64(video_path)
            log("  ✓ Done")

            try: os.remove(video_path)
            except Exception: pass

            return {
                "status":         "success",
                "job_id":         job_id,
                "video_base64":   b64,
                "video_filename": "%s.mp4" % job_id,
                "video_size_mb":  round(mb, 2),
                "voice_effect":   voice_effect,
                "note":           "Decode video_base64 with base64.b64decode() to get the mp4.",
            }

    except Exception as exc:
        return error_response(job_id, "top_level", exc)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
