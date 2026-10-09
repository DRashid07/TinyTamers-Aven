"""Landmark arrays -> fixed-size feature matrix for the classifier.

Owner: C (Backend/LLM). See CONTRACT.md "Features". numpy only.

Columns of X (192 per frame, every x, y pair is point-major: x0, y0, x1, y1, ...):
  0-21     pose POSE_IDX, body x, y
  22-63    left hand, body x, y          64-105   left hand, hand-local x, y
  106-147  right hand, body x, y         148-189  right hand, hand-local x, y
  190      left_present                  191      right_present
"""
import numpy as np

from pose.normalize import assign_hands, to_body

FEATURE_VERSION = 1
POSE_IDX = [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]
N_FEATURES = 192


def _hand_block(body):
    """Body x, y (T,21,2), NaN = missing -> (T,84) [body x, y | hand-local x, y], present (T,) bool."""
    size = np.linalg.norm(body[:, 9] - body[:, 0], axis=1)
    present = ~np.isnan(body).any(axis=(1, 2)) & (size > 1e-6)
    local = (body - body[:, :1]) / np.where(present, size, 1.0)[:, None, None]
    block = np.concatenate([body.reshape(len(body), 42), local.reshape(len(body), 42)], axis=1)
    block[~present] = 0.0
    return block, present


def arrays_to_features(pose, hands, t, w, h, T=32):
    """Return (X float32 (T, 192), info dict with n_frames, valid_ratio, hand_ratio, duration_ms).

    Invalid frames are dropped, the rest is resampled to T frames by linear interpolation over t.
    hand_ratio = share of valid frames with at least one hand; no valid frame -> X is all zeros.
    """
    n = len(t)
    t = np.asarray(t, dtype=np.float64)
    left, right = assign_hands(pose, hands)
    body_pose, valid = to_body(pose, pose[:, POSE_IDX], w, h)
    left_block, left_present = _hand_block(to_body(pose, left, w, h)[0])
    right_block, right_present = _hand_block(to_body(pose, right, w, h)[0])
    frames = np.concatenate([body_pose.reshape(n, 2 * len(POSE_IDX)), left_block, right_block,
                             left_present[:, None], right_present[:, None]], axis=1)

    X = np.zeros((T, N_FEATURES), np.float32)
    if valid.any():
        tv, fv = t[valid], frames[valid]
        grid = np.linspace(tv[0], tv[-1], T)
        X = np.stack([np.interp(grid, tv, column) for column in fv.T], axis=1)
        X = np.nan_to_num(X).astype(np.float32)
    info = {
        "n_frames": int(n),
        "valid_ratio": float(valid.mean()) if n else 0.0,
        "hand_ratio": float((left_present | right_present)[valid].mean()) if valid.any() else 0.0,
        "duration_ms": float(t[-1] - t[0]) if n else 0.0,
    }
    return X, info
