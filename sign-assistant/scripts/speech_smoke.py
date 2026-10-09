"""Smoke-test Azerbaijani text-to-speech: 5 test sentences -> scripts/out/*.mp3 (gitignored).

Owner: D (Direction B/Speech/Eval). Run from sign-assistant/:
    python -m scripts.speech_smoke [--engine auto|azure|mms] [--mms-model ID_OR_FOLDER]
azure: Azure Speech REST TTS, voices az-AZ-BanuNeural and az-AZ-BabekNeural, SSML, mp3; AZURE_SPEECH_KEY and
       AZURE_SPEECH_REGION from .env. Text goes to Microsoft's cloud.
mms:   Meta MMS-TTS facebook/mms-tts-azj-script_latin, runs locally. Licence CC BY-NC 4.0 (non-commercial).
       Needs `transformers`, which is NOT in requirements.txt; mp3 needs ffmpeg (else .wav is kept).
auto:  azure, and mms only if azure is not configured or fails.
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scripts" / "out"
SENTENCES = [
    "Sabah həkimə getmək istəyirəm.",
    "Başım ağrıyır.",
    "Növbəm nə vaxtdır?",
    "Mənə kömək edin.",
    "Sənədlərimi gətirmişəm.",
]
VOICES = ["az-AZ-BanuNeural", "az-AZ-BabekNeural"]
MMS_MODEL = "facebook/mms-tts-azj-script_latin"


def ssml(text, voice):
    return (f"<speak version='1.0' xml:lang='az-AZ'><voice xml:lang='az-AZ' name='{voice}'>"
            f"{escape(text)}</voice></speak>")


def azure_tts(text, voice, key, region):
    """One sentence -> mp3 bytes from the Azure Speech REST API (raises on any HTTP or network error)."""
    request = urllib.request.Request(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
        data=ssml(text, voice).encode("utf-8"),
        headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/ssml+xml",
                 "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
                 "User-Agent": "sign-assistant-speech-smoke"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read()


def run_azure():
    key, region = os.environ.get("AZURE_SPEECH_KEY"), os.environ.get("AZURE_SPEECH_REGION")
    if not key or not region:
        print("azure: not configured (AZURE_SPEECH_KEY / AZURE_SPEECH_REGION empty in .env)")
        return False
    try:
        for n, text in enumerate(SENTENCES, 1):
            for voice in VOICES:
                path = OUT / f"azure_{voice}_{n}.mp3"
                path.write_bytes(azure_tts(text, voice, key, region))
                print(f"azure: {path.name}  {path.stat().st_size} B  {text}")
        return True
    except (urllib.error.URLError, OSError) as err:
        print(f"azure: FAILED ({err})")
        return False


def write_wav(path, samples, rate):
    import numpy as np

    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def run_mms(model_id=MMS_MODEL):
    try:
        import torch
        from transformers import AutoTokenizer, VitsModel
    except ImportError:
        print("mms: `transformers` is not installed (pip install transformers); skipped")
        return False
    tokenizer, model = AutoTokenizer.from_pretrained(model_id), VitsModel.from_pretrained(model_id).eval()
    ffmpeg = shutil.which("ffmpeg")
    for n, text in enumerate(SENTENCES, 1):
        torch.manual_seed(n)  # VITS samples durations and noise: fixed seed = same audio every run
        with torch.no_grad():
            samples = model(**tokenizer(text, return_tensors="pt")).waveform[0].numpy()
        wav = OUT / f"mms_{n}.wav"
        write_wav(wav, samples, model.config.sampling_rate)
        path = wav
        if ffmpeg and subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(wav), "-codec:a", "libmp3lame",
                                      "-b:a", "64k", str(wav.with_suffix(".mp3"))]).returncode == 0:
            wav.unlink()
            path = wav.with_suffix(".mp3")
        print(f"mms: {path.name}  {len(samples) / model.config.sampling_rate:.1f} s  {text}")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description="Azerbaijani TTS smoke test.")
    p.add_argument("--engine", choices=["auto", "azure", "mms"], default="auto")
    p.add_argument("--mms-model", default=MMS_MODEL, help="Hugging Face id or a local folder of MMS files")
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    OUT.mkdir(parents=True, exist_ok=True)

    ok = False
    if a.engine in ("auto", "azure"):
        ok = run_azure()
    if a.engine == "mms" or (a.engine == "auto" and not ok):
        ok = run_mms(a.mms_model) or ok
    print(f"files in {OUT}" if ok else "no audio produced")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
