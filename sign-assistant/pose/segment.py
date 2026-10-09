"""Cut one sign out of a frame stream by raised hands, live (Segmenter) and offline.

Owner: C (Backend/LLM). Parameters in pose/segment_config.json. See CONTRACT.md "Segmentation".
A segment runs from the first to the last "up" frame; live and offline share one state machine.
"""
import json
from pathlib import Path

import numpy as np

from pose.normalize import frames_to_arrays, to_body

CONFIG_PATH = Path(__file__).with_name("segment_config.json")


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def raised(pose, hands, w, h, raise_y):
    """(T,) bool: some present hand has a body-coordinate wrist y < raise_y (invalid frames: False)."""
    wrist_y = to_body(pose, hands[:, :, 0], w, h)[0][..., 1]
    return (np.nan_to_num(wrist_y, nan=np.inf) < raise_y).any(axis=1)


def status(duration_ms, cfg):
    if duration_ms < cfg["min_ms"]:
        return "too_short"
    return "too_long" if duration_ms > cfg["max_ms"] else "ok"


class _Cutter:
    """Start after start_frames up frames in a row, end after end_frames not-up frames in a row."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.active = False
        self.run = 0  # consecutive up frames while idle, consecutive not-up frames while active
        self.first = self.last = None  # frame indices of the first and the last up frame

    def step(self, i, up):
        """Feed frame i; return None, "start" or "end"."""
        if not self.active:
            self.run = self.run + 1 if up else 0
            if self.run < self.cfg["start_frames"]:
                return None
            self.active, self.run, self.first, self.last = True, 0, i - self.cfg["start_frames"] + 1, i
            return "start"
        if up:
            self.run, self.last = 0, i
            return None
        self.run += 1
        if self.run < self.cfg["end_frames"]:
            return None
        self.active, self.run = False, 0
        return "end"


class Segmenter:
    """Live: push frame dicts one at a time (the session header is not a frame)."""

    def __init__(self, w, h, cfg=None):
        self.w, self.h = w, h
        self.cfg = cfg or load_config()
        self.cut = _Cutter(self.cfg)
        self.buf, self.buf0, self.n = [], 0, 0  # buffered frames, index of buf[0], frames seen

    @property
    def signing(self):
        return self.cut.active

    def push(self, frame_dict):
        """Return None | {"event": "start"} | {"event": "end", "frames": [...], "status": ...}."""
        if frame_dict.get("type", "frame") != "frame":
            return None
        pose, hands, _ = frames_to_arrays([frame_dict])
        up = bool(raised(pose, hands, self.w, self.h, self.cfg["raise_y"])[0])
        i, self.n = self.n, self.n + 1
        self.buf.append(frame_dict)
        event = self.cut.step(i, up)
        if event == "start":
            return {"event": "start"}
        if event == "end":
            frames = self.buf[self.cut.first - self.buf0:self.cut.last - self.buf0 + 1]
            self.buf, self.buf0 = [], i + 1
            duration = frames[-1]["t"] - frames[0]["t"]
            return {"event": "end", "frames": frames, "status": status(duration, self.cfg)}
        if not self.cut.active and len(self.buf) > self.cfg["start_frames"]:
            drop = len(self.buf) - self.cfg["start_frames"]  # idle: keep only a possible start run
            self.buf, self.buf0 = self.buf[drop:], self.buf0 + drop
        return None


def segment_offline(pose, hands, t, w, h, cfg=None):
    """Return (start_idx, end_idx) from the first start to the last end, or None.

    end_idx is exclusive, so pose[start_idx:end_idx] is the cut. A sign still raised at the end of
    the recording is closed at its last up frame.
    """
    cfg = cfg or load_config()
    cut, spans = _Cutter(cfg), []
    for i, up in enumerate(raised(pose, hands, w, h, cfg["raise_y"])):
        if cut.step(i, up) == "end":
            spans.append((cut.first, cut.last))
    if cut.active:
        spans.append((cut.first, cut.last))
    return (spans[0][0], spans[-1][1] + 1) if spans else None
