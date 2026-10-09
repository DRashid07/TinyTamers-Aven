"""Extract MediaPipe landmarks for each vocabulary video in data/index.csv -> data/landmarks/<video_id>.npz.

Owner: A (Data/ML). Run from sign-assistant/:
    python -m pose.extract_dataset --limit-classes 3
    python -m pose.extract_dataset --shard 1/2        # laptop 1 of 2 (the other runs --shard 2/2)
Uses the browser's .task files (web/models/) in VIDEO mode and builds the arrays with
pose.normalize.frames_to_arrays, so the .npz files match the live path. Existing files are skipped.
"""
import argparse
import csv
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from pose.normalize import frames_to_arrays

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "web" / "models"
OUT_DIR = ROOT / "data" / "landmarks"
ERRORS_CSV = ROOT / "data" / "reports" / "extract_errors.csv"

_models = {}  # per worker: .task file bytes, read once


def select_rows(rows, vocab, camera="front", limit_classes=None, shard=(1, 1)):
    """Rows whose label is in the first limit_classes vocab entries and whose camera matches
    ("any" = every camera); shard (i, n) keeps every n-th row by video_id, starting at the i-th."""
    labels = {v["dataset_label"] for v in vocab[:limit_classes]}
    rows = sorted((r for r in rows if r["dataset_label"] in labels and camera in ("any", r["camera"])),
                  key=lambda r: r["video_id"])
    i, n = shard
    return rows[i - 1::n]


def _landmarkers():
    """Fresh VIDEO-mode landmarkers, so tracking never carries over from the previous clip."""
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (HandLandmarker, HandLandmarkerOptions, PoseLandmarker,
                                               PoseLandmarkerOptions, RunningMode)
    if not _models:
        for name in ("pose_landmarker_lite.task", "hand_landmarker.task"):
            _models[name] = (MODELS / name).read_bytes()
    pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=_models["pose_landmarker_lite.task"]),
        running_mode=RunningMode.VIDEO, num_poses=1))
    hands = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=_models["hand_landmarker.task"]),
        running_mode=RunningMode.VIDEO, num_hands=2))
    return pose, hands


def extract(row, target_fps):
    """One video -> .npz. Returns a stats dict, with "error" set on failure."""
    import mediapipe as mp

    stats = {"video_id": row["video_id"], "video_path": row["video_path"], "frames": 0, "pose": 0, "hand": 0}
    out = OUT_DIR / f"{row['video_id']}.npz"
    capture = cv2.VideoCapture(str(ROOT / row["video_path"]))
    try:
        if not capture.isOpened():
            raise OSError("OpenCV cannot open the video")
        fps = capture.get(cv2.CAP_PROP_FPS) or float(row["fps"] or 30)
        step = max(1, round(fps / target_fps))
        w, h = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        pose_lm, hand_lm = _landmarkers()
        frames, index = [], 0
        with pose_lm, hand_lm:
            while capture.grab():
                if index % step == 0:
                    ok, bgr = capture.retrieve()
                    if not ok:
                        break
                    t = index * 1000.0 / fps  # ms from the original frame index
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    p = pose_lm.detect_for_video(image, int(round(t)))
                    hd = hand_lm.detect_for_video(image, int(round(t)))
                    pose = [[l.x, l.y, l.z, l.visibility] for l in p.pose_landmarks[0]] if p.pose_landmarks else None
                    frames.append({
                        "type": "frame", "t": t, "pose": pose,
                        "hands": [[[l.x, l.y, l.z] for l in hand] for hand in hd.hand_landmarks],
                    })
                index += 1
        if not frames:
            raise ValueError("no frames decoded")
        pose, hands, t = frames_to_arrays(frames)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, pose=pose, hands=hands, t=t, w=w, h=h)
        os.replace(tmp, out)  # a killed run never leaves a half-written .npz behind
        stats.update(frames=len(frames), pose=sum(f["pose"] is not None for f in frames),
                     hand=sum(bool(f["hands"]) for f in frames))
    except Exception as err:  # noqa: BLE001 - every failure is logged, the run goes on
        stats["error"] = f"{type(err).__name__}: {err}"
    finally:
        capture.release()
    return stats


def _extract_star(args):
    return extract(*args)


def log_errors(failed):
    new = not ERRORS_CSV.exists()
    with open(ERRORS_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["time", "video_id", "video_path", "error"])
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        writer.writerows([stamp, s["video_id"], s["video_path"], s["error"]] for s in failed)


def main(argv=None):
    p = argparse.ArgumentParser(description="Extract landmarks for the vocabulary videos.")
    p.add_argument("--target-fps", type=float, default=15, help="process every k-th frame, about this rate")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit-classes", type=int, default=None, help="only the first N classes of vocab.json")
    p.add_argument("--shard", default="1/1", help="i/n: every n-th video starting at the i-th (1-based)")
    p.add_argument("--camera", default="front", help="front, side, unknown or any")
    p.add_argument("--index", type=Path, default=ROOT / "data" / "index.csv")
    p.add_argument("--vocab", type=Path, default=ROOT / "data" / "vocab.json")
    a = p.parse_args(argv)
    i, n = (int(x) for x in a.shard.split("/"))
    if not 1 <= i <= n:
        p.error("--shard must be i/n with 1 <= i <= n")

    with open(a.index, encoding="utf-8", newline="") as f:
        index_rows = list(csv.DictReader(f))
    vocab = json.loads(a.vocab.read_text(encoding="utf-8"))
    rows = select_rows(index_rows, vocab, a.camera, a.limit_classes, (i, n))
    todo = [r for r in rows if not (OUT_DIR / f"{r['video_id']}.npz").exists()]
    print(f"{len(rows)} videos selected ({len(vocab[:a.limit_classes])} classes, camera={a.camera}, "
          f"shard {i}/{n}); {len(rows) - len(todo)} already done; {len(todo)} to do with {a.workers} workers")
    if not todo:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ERRORS_CSV.parent.mkdir(parents=True, exist_ok=True)

    start, done, frames, with_pose, with_hand, failed = time.time(), 0, 0, 0, 0, []
    report_every = max(1, len(todo) // 20)
    with Pool(a.workers) as pool:
        for s in pool.imap_unordered(_extract_star, [(r, a.target_fps) for r in todo], chunksize=4):
            done += 1
            frames, with_pose, with_hand = frames + s["frames"], with_pose + s["pose"], with_hand + s["hand"]
            if "error" in s:
                failed.append(s)
            if done % report_every == 0 or done == len(todo):
                sec = time.time() - start
                eta = (len(todo) - done) / done * sec / 60
                print(f"{done}/{len(todo)} videos | {done / sec * 60:.0f} videos/min | {frames / sec:.0f} frames/s"
                      f" | {len(failed)} failed | ETA {eta:.1f} min", flush=True)
    if failed:
        log_errors(failed)
    sec = time.time() - start
    print(f"done in {sec / 60:.1f} min: {done / sec * 60:.0f} videos/min, {frames} frames, "
          f"pose in {with_pose / max(frames, 1):.1%}, >=1 hand in {with_hand / max(frames, 1):.1%}, "
          f"{len(failed)} failed" + (f" (see {ERRORS_CSV.relative_to(ROOT)})" if failed else ""))


if __name__ == "__main__":
    main()
