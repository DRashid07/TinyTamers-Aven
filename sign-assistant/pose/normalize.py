"""Frame dicts -> landmark arrays, hands -> signer's left/right, and body coordinates.

Owner: C (Backend/LLM). See CONTRACT.md "Arrays" and "Normalisation". numpy only.
"""
import numpy as np

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16


def frames_to_arrays(frames):
    """Return pose (T,33,4) float32, hands (T,2,21,3) float32 (NaN = missing, unordered), t (T,) ms."""
    frames = [f for f in frames if f.get("type", "frame") == "frame"]
    pose = np.full((len(frames), 33, 4), np.nan, np.float32)
    hands = np.full((len(frames), 2, 21, 3), np.nan, np.float32)
    for i, f in enumerate(frames):
        if f.get("pose") is not None:
            pose[i] = f["pose"]
        for k, hand in enumerate((f.get("hands") or [])[:2]):
            hands[i, k] = hand
    t = np.array([f["t"] for f in frames], dtype=np.float64)
    return pose, hands, t


def assign_hands(pose, hands):
    """Give each detected hand to the nearer pose wrist; return left (T,21,3), right (T,21,3).

    Pose 15 is the signer's left wrist, 16 the right. With two hands, the pairing with the smaller
    total wrist distance wins, so they never both land on one side. MediaPipe handedness is never used.
    """
    left = np.full((len(hands), 21, 3), np.nan, np.float32)
    right = left.copy()
    for i in range(len(hands)):
        found = [hand for hand in hands[i] if not np.isnan(hand).any()]
        if not found:
            continue
        # distance from each hand's wrist (point 0) to the pose wrists [left, right]; NaN -> inf
        d = np.array([[np.hypot(*(hand[0, :2] - pose[i, w, :2])) for w in (L_WRIST, R_WRIST)]
                      for hand in found])
        d = np.nan_to_num(d, nan=np.inf)
        if np.isinf(d).all():
            continue  # no pose wrists: the hand cannot be placed
        if len(found) == 1:
            (left if d[0, 0] <= d[0, 1] else right)[i] = found[0]
        elif d[0, 0] + d[1, 1] <= d[0, 1] + d[1, 0]:
            left[i], right[i] = found
        else:
            right[i], left[i] = found
    return left, right


def to_body(pose, points, w, h):
    """Contract "Normalisation" -> (body x, y (T,N,2) float32, valid (T,) bool).

    points (T,N,>=2) are image-normalised landmarks of the same frames as pose (pose itself or hands).
    x is multiplied by w / h, then body = (p - shoulder midpoint) / shoulder distance; y points down.
    A frame is invalid (all NaN) if a shoulder is NaN, its visibility < 0.5, or the shoulders coincide.
    """
    aspect = np.array([w / h, 1.0], dtype=np.float32)
    shoulders = pose[:, [L_SHOULDER, R_SHOULDER], :2] * aspect
    origin = shoulders.mean(axis=1)
    scale = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    valid = (~np.isnan(shoulders).any(axis=(1, 2))
             & (pose[:, [L_SHOULDER, R_SHOULDER], 3] >= 0.5).all(axis=1)
             & (scale > 1e-6))
    scale = np.where(valid, scale, 1.0)
    body = (points[..., :2] * aspect - origin[:, None]) / scale[:, None, None]
    body[~valid] = np.nan
    return body.astype(np.float32), valid
