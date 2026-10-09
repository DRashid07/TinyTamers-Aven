"""Landmark arrays -> fixed-size feature matrix for the classifier.

Owner: C (Backend/LLM). See CONTRACT.md "Features".
"""

FEATURE_VERSION = 1


def arrays_to_features(pose, hands, t, w, h, T=32):
    """Return (X float32 (T, 192), info dict with n_frames, valid_ratio, hand_ratio, duration_ms)."""
    raise NotImplementedError
