"""FastAPI app: API routers, static web UI, sign clips and a health check.

Owner: C (Backend/LLM). Run from sign-assistant/: uvicorn api.main:app
"""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
CLIPS_DIR = ROOT / "data" / "clips"

# Load .env before importing routers, so they can read config at import time.
load_dotenv(ROOT / ".env")

from api import compose, routes_recognise, speech, text_to_signs  # noqa: E402

CLIPS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AzSL sign assistant (prototype)")
app.include_router(routes_recognise.router)
app.include_router(compose.router)
app.include_router(text_to_signs.router)
app.include_router(speech.router)


@app.get("/health")
def health():
    return {"ok": True}


# Mounts last: "/" catches every path not matched above.
app.mount("/clips", StaticFiles(directory=CLIPS_DIR), name="clips")
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
