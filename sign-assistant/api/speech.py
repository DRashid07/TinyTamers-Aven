"""POST /tts: text -> audio/mpeg via the Azure Speech REST API; 503 if not configured.

Owner: D (Direction B/Speech/Eval). See CONTRACT.md "API".
Needs AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env (AZURE_SPEECH_VOICE optional). The text is sent
to Microsoft Azure and is never logged. Audio is cached in memory by text hash (LRU, CACHE_SIZE items).

Frontend (B), only inside a click handler so the browser allows playback:
    fetch("/tts", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text})})
      .then((r) => (r.ok ? r.blob() : Promise.reject(r.status)))
      .then((blob) => { const url = URL.createObjectURL(blob); const audio = new Audio(url);
                        audio.onended = () => URL.revokeObjectURL(url); audio.play(); });
503 = TTS not configured (show "Səsləndirmə qoşulmayıb"), 422 = empty or over 300 characters,
502 / 504 = Azure failed or did not answer in time.
"""
import hashlib
import logging
import os
import re
import threading
from collections import OrderedDict
from xml.sax.saxutils import escape

import httpx
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

DEFAULT_VOICE = "az-AZ-BanuNeural"  # Azerbaijani female voice; az-AZ-BabekNeural is the male one
OUTPUT_FORMAT = "audio-24khz-48kbitrate-mono-mp3"
TIMEOUT_S = 8.0
MAX_CHARS = 300
CACHE_SIZE = 100

log = logging.getLogger("uvicorn.error")
router = APIRouter()
_cache = OrderedDict()  # sha256(voice + text) -> mp3 bytes, oldest first
_lock = threading.Lock()  # sync endpoints run in a thread pool


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


def ssml(text, voice):
    voice = escape(voice, {"'": "&apos;"})
    return f"<speak version='1.0' xml:lang='az-AZ'><voice name='{voice}'>{escape(text)}</voice></speak>"


def synthesize(text, key, region, voice):
    """Azure REST text to speech -> MP3 bytes. HTTPException 504 on timeout, 502 on any other failure."""
    try:
        response = httpx.post(
            f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
            content=ssml(text, voice).encode("utf-8"),
            headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/ssml+xml",
                     "X-Microsoft-OutputFormat": OUTPUT_FORMAT, "User-Agent": "sign-assistant"},
            timeout=TIMEOUT_S,
        )
    except httpx.TimeoutException:
        log.warning("TTS: Azure did not answer within %.0f s", TIMEOUT_S)
        raise HTTPException(504, f"Azure TTS did not answer within {TIMEOUT_S:.0f} s") from None
    except httpx.HTTPError as err:
        log.warning("TTS: Azure not reachable (%s)", type(err).__name__)
        raise HTTPException(502, "Azure TTS is not reachable") from None
    if response.status_code != 200 or not response.content:
        log.warning("TTS: Azure answered HTTP %s", response.status_code)
        raise HTTPException(502, f"Azure TTS failed (HTTP {response.status_code})")
    return response.content


@router.post("/tts")
def tts(body: TTSRequest):
    """{"text"} (1-300 characters) -> audio/mpeg. 422 for a bad text, 503 without Azure settings,
    502/504 when Azure fails."""
    key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
    region = os.environ.get("AZURE_SPEECH_REGION", "").strip().lower()
    if not key or not region:
        raise HTTPException(503, "TTS is not configured: set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env")
    if not re.fullmatch(r"[a-z0-9]+", region):
        raise HTTPException(503, "TTS is not configured: AZURE_SPEECH_REGION must look like 'westeurope'")
    text = body.text.strip()
    if not text:
        raise HTTPException(422, "text is empty")
    voice = os.environ.get("AZURE_SPEECH_VOICE", "").strip() or DEFAULT_VOICE

    digest = hashlib.sha256(f"{voice}\n{text}".encode("utf-8")).hexdigest()
    with _lock:
        audio = _cache.get(digest)
        if audio is not None:
            _cache.move_to_end(digest)
    if audio is None:
        audio = synthesize(text, key, region, voice)
        with _lock:
            _cache[digest] = audio
            while len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
    return Response(content=audio, media_type="audio/mpeg")
