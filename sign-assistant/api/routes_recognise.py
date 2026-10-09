"""WS /ws/recognise, POST /recognise, GET /vocab, GET /model-info.

Owner: C (Backend/LLM). See CONTRACT.md "API" and "Result".
Pipeline per sign: segment -> arrays_to_features -> Predictor -> decide. Landmarks are never stored;
the log keeps only duration, latency and the result.
"""
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from model.abstain import abstain, decide
from model.predictor import Predictor
from pose.features import arrays_to_features
from pose.normalize import frames_to_arrays
from pose.segment import Segmenter, load_config, segment_offline, status

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "data" / "vocab.json"
PREDICTOR = Predictor(ROOT / os.environ.get("MODEL_DIR", "model/artifacts"))

log = logging.getLogger("uvicorn.error")
router = APIRouter()


def load_vocab():
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


def check_frame(frame):
    """Raise ValueError unless frame has the contract shape (numpy would silently broadcast a wrong one)."""
    float(frame["t"])
    if frame.get("pose") is not None and np.shape(frame["pose"]) != (33, 4):
        raise ValueError("pose must be 33 x [x, y, z, vis] or null")
    hands = frame.get("hands") or []
    if len(hands) > 2 or any(np.shape(hand) != (21, 3) for hand in hands):
        raise ValueError("hands must be 0-2 x 21 x [x, y, z]")


def recognise(frames, w, h, segment_status="ok"):
    """One cut sign (frame dicts) -> Result. too_short/too_long and a missing model abstain first."""
    start = time.perf_counter()
    if segment_status != "ok":
        result = abstain(segment_status)
    elif not PREDICTOR.loaded:
        result = abstain("model_not_loaded")
    else:
        try:
            pose, hands, t = frames_to_arrays(frames)
            X, info = arrays_to_features(pose, hands, t, w, h, T=PREDICTOR.config["T"])
            result = decide(PREDICTOR.probs(X), info, PREDICTOR.config, load_vocab())
        except Exception:  # noqa: BLE001 - a model error must abstain, never guess
            log.exception("recognise failed")
            result = abstain("model_not_loaded")
    duration = frames[-1]["t"] - frames[0]["t"] if frames else 0.0
    outcome = (f"ok {result['id']} {result['confidence']}" if result["status"] == "ok"
               else f"abstain {result['reason']}")
    log.info("segment: %d frames, %.0f ms, latency %.1f ms -> %s", len(frames), duration,
             (time.perf_counter() - start) * 1000, outcome)
    return result


@router.get("/vocab")
def vocab():
    return load_vocab()


@router.get("/model-info")
def model_info():
    cfg = PREDICTOR.config or {}
    return {"loaded": PREDICTOR.loaded, "arch": cfg.get("arch"), "n_classes": len(cfg.get("classes", [])),
            "tau": cfg.get("tau"), "margin": cfg.get("margin"), "temperature": cfg.get("temperature")}


class Sequence(BaseModel):
    """POST /recognise body. Extra keys (a record-mode file's "id", "signer") are ignored."""
    w: int
    h: int
    frames: list[dict]
    segment: bool = True


@router.post("/recognise")
def recognise_sequence(body: Sequence):
    """Full recorded sequence -> Result. segment=true cuts the sign like the live path
    (no raised hands -> no_hands); segment=false uses every frame as one sign."""
    w, h = body.w, body.h
    frames = [f for f in body.frames if f.get("type", "frame") == "frame"]
    try:
        if w <= 0 or h <= 0:
            raise ValueError("w and h must be positive")
        for frame in frames:
            check_frame(frame)
        pose, hands, t = frames_to_arrays(frames)
    except (ValueError, KeyError, TypeError) as err:
        raise HTTPException(422, f"bad frames: {err}") from err
    if not body.segment:
        return recognise(frames, w, h)
    cfg = load_config()
    span = segment_offline(pose, hands, t, w, h, cfg)
    if span is None:
        return recognise(frames, w, h, "no_hands")
    frames = frames[span[0]:span[1]]
    return recognise(frames, w, h, status(frames[-1]["t"] - frames[0]["t"], cfg))


@router.websocket("/ws/recognise")
async def ws_recognise(ws: WebSocket):
    """Header {"type": "start", "w", "h"} then frames. Sends state on start/end and a result per sign."""
    await ws.accept()
    segmenter = None
    try:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
                if msg.get("type") == "start":
                    if int(msg["w"]) <= 0 or int(msg["h"]) <= 0:
                        raise ValueError("w and h must be positive")
                    segmenter = Segmenter(int(msg["w"]), int(msg["h"]))
                    continue
                if segmenter is None or msg.get("type") != "frame":
                    continue
                check_frame(msg)
                event = segmenter.push(msg)
            except (ValueError, KeyError, TypeError) as err:  # a bad message is dropped, the session goes on
                log.warning("ws/recognise: dropped a bad message (%s)", err)
                continue
            if event is None:
                continue
            await ws.send_json({"type": "state", "signing": event["event"] == "start"})
            if event["event"] == "end":
                result = await run_in_threadpool(recognise, event["frames"], segmenter.w, segmenter.h,
                                                  event["status"])
                await ws.send_json({"type": "result", **result})
    except WebSocketDisconnect:
        pass
