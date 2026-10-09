"""POST /tts: text -> audio/mpeg via Azure Speech; 503 if not configured.

Owner: D (Direction B/Speech/Eval). See CONTRACT.md "API".
"""
from fastapi import APIRouter

router = APIRouter()
