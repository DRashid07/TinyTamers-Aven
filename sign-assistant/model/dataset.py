"""Training data: landmark files + vocab + group splits -> feature tensors, and train-time augmentation.

Owner: A (Data/ML).
CONTRACT v2: AzSLD Words clips are used whole; team recordings (team_* groups) are cut with
pose.segment.segment_offline, falling back to the whole clip when it finds no sign. Features of every
train/val video are cached in data/landmarks/features_v1.npz. Test groups are never read here:
the test split belongs to `python -m eval.evaluate --split test --final`.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pose.features import FEATURE_VERSION, arrays_to_features
from pose.segment import load_config, segment_offline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "landmarks" / f"features_v{FEATURE_VERSION}.npz"

# Column blocks of a feature frame (pose/features.py): body x, y blocks, hand-local blocks, presence flags.
POSE_BODY = slice(0, 22)
HANDS = [(slice(22, 64), slice(64, 106), 190), (slice(106, 148), slice(148, 190), 191)]  # left, right


def load_vocab(data_dir=DATA):
    return json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))


def candidate_rows(data_dir, vocab, camera):
    """index.csv rows of the vocabulary with the camera (any = every camera), test groups removed."""
    splits = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
    labels, test = {v["dataset_label"] for v in vocab}, set(splits["test"])
    with open(data_dir / "index.csv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset_label"] in labels and camera in ("any", r["camera"])]
    return [r for r in rows if r["group"] not in test], splits


def build_features(rows, vocab, landmarks_dir, T=32):
    """Rows -> dict of arrays (X, y, video_id, group, valid_ratio, hand_ratio) and a report of
    missing landmark files per label and segment_offline fallbacks (team recordings only)."""
    class_of = {v["dataset_label"]: i for i, v in enumerate(vocab)}
    cfg = load_config()
    out = {k: [] for k in ("X", "y", "video_id", "group", "valid_ratio", "hand_ratio")}
    report = {"missing": Counter(), "cut": 0, "fallback": 0}
    for r in rows:
        path = landmarks_dir / f"{r['video_id']}.npz"
        if not path.exists():
            report["missing"][r["dataset_label"]] += 1
            continue
        z = np.load(path)
        pose, hands, t, w, h = z["pose"], z["hands"], z["t"], int(z["w"]), int(z["h"])
        if r["group"].startswith("team_"):
            span = segment_offline(pose, hands, t, w, h, cfg)
            if span is None:
                report["fallback"] += 1
            else:
                report["cut"] += 1
                pose, hands, t = pose[span[0]:span[1]], hands[span[0]:span[1]], t[span[0]:span[1]]
        x, info = arrays_to_features(pose, hands, t, w, h, T=T)
        for key, value in (("X", x), ("y", class_of[r["dataset_label"]]), ("video_id", r["video_id"]),
                           ("group", r["group"]), ("valid_ratio", info["valid_ratio"]),
                           ("hand_ratio", info["hand_ratio"])):
            out[key].append(value)
    data = {k: np.array(v) for k, v in out.items()}
    data["X"] = data["X"].astype(np.float32).reshape(-1, T, 192)
    return data, report


def cached_features(rows, vocab, data_dir=DATA, cache=CACHE, T=32, rebuild=False):
    """Features for rows, from the cache when it holds exactly these videos, classes and T."""
    present = {r["video_id"] for r in rows if (data_dir / "landmarks" / f"{r['video_id']}.npz").exists()}
    classes = [v["id"] for v in vocab]
    if cache.exists() and not rebuild:
        z = np.load(cache)
        if z["classes"].tolist() == classes and int(z["T"]) == T and set(z["video_id"].tolist()) == present:
            data = {k: z[k] for k in ("X", "y", "video_id", "group", "valid_ratio", "hand_ratio")}
            data["X"] = data["X"].astype(np.float32)
            return data, {"missing": Counter(r["dataset_label"] for r in rows if r["video_id"] not in present),
                          "cut": int(z["cut"]), "fallback": int(z["fallback"]), "cached": True}
    data, report = build_features(rows, vocab, data_dir / "landmarks", T)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{**data, "X": data["X"].astype(np.float16)}, classes=np.array(classes), T=T,
             cut=report["cut"], fallback=report["fallback"])
    return data, {**report, "cached": False}


def load_split(split, camera="front", data_dir=DATA, cache=CACHE, T=32, rebuild=False):
    """"train" or "val" -> (X (N, T, 192) float32, y (N,) class indices, report).

    train = the train groups plus videos with an empty group (train-only, CONTRACT v2).
    """
    if split == "test":
        raise ValueError("the test split is read only by python -m eval.evaluate --split test --final")
    if split not in ("train", "val"):
        raise ValueError(f"unknown split {split!r}")
    vocab = load_vocab(data_dir)
    rows, splits = candidate_rows(data_dir, vocab, camera)
    if not rows:
        raise ValueError(f"no {camera!r}-camera videos of the vocabulary in index.csv (AzSLD needs camera='any')")
    data, report = cached_features(rows, vocab, data_dir, cache, T, rebuild)
    groups = set(splits[split]) | ({""} if split == "train" else set())
    keep = np.isin(data["group"], list(groups))
    return data["X"][keep], data["y"][keep].astype(np.int64), report


def augment(x, rng):
    """One train sample (T, 192) -> new array: rotation up to 10 degrees, scale 0.9-1.1 and shift up to
    0.05 in body coordinates, Gaussian noise 0.01, temporal crop 85-100% with 10% random frame drop,
    resampled back to T. Missing hands stay zero. Never mirrors left/right."""
    T = len(x)
    x = x.copy()
    angle = np.deg2rad(rng.uniform(-10, 10))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    scale, shift = rng.uniform(0.9, 1.1), rng.uniform(-0.05, 0.05, size=2).astype(np.float32)
    x[:, POSE_BODY] = ((x[:, POSE_BODY].reshape(T, -1, 2) @ rot.T) * scale + shift).reshape(T, -1)
    x[:, POSE_BODY] += rng.normal(0, 0.01, size=(T, 22))
    for body, local, flag in HANDS:
        present = (x[:, flag] > 0)[:, None]
        moved = ((x[:, body].reshape(T, -1, 2) @ rot.T) * scale + shift).reshape(T, -1)
        turned = (x[:, local].reshape(T, -1, 2) @ rot.T).reshape(T, -1)
        x[:, body] = np.where(present, moved + rng.normal(0, 0.01, size=moved.shape), 0.0)
        x[:, local] = np.where(present, turned + rng.normal(0, 0.01, size=turned.shape), 0.0)

    span = rng.uniform(0.85, 1.0) * (T - 1)
    start = rng.uniform(0, (T - 1) - span)
    keep = np.flatnonzero(rng.random(T) >= 0.1)
    if len(keep) < 2:
        keep = np.arange(T)
    pos = np.interp(np.linspace(start, start + span, T), keep, np.arange(len(keep)))
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(keep) - 1)
    w = (pos - i0)[:, None]
    return (x[keep[i0]] * (1 - w) + x[keep[i1]] * w).astype(np.float32)
