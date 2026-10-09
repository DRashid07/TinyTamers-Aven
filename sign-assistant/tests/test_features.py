"""Tests for pose/features.py on synthetic frames.

Owner: C (Backend/LLM).
"""
import numpy as np

from pose.features import FEATURE_VERSION, POSE_IDX, arrays_to_features
from pose.normalize import frames_to_arrays
from tests.fake_pose import FPS_MS, H, UP, W, fake_frame, raise_both, sequence

LEFT_LOCAL, RIGHT_LOCAL, LEFT_PRESENT, RIGHT_PRESENT = 64, 148, 190, 191


def features(frames, **kw):
    return arrays_to_features(*frames_to_arrays(frames), W, H, **kw)


def test_shape_dtype_and_no_nan():
    X, info = features(sequence((10, {}), (20, raise_both()), (10, {})))
    assert FEATURE_VERSION == 1
    assert X.shape == (32, 192) and X.dtype == np.float32
    assert not np.isnan(X).any()
    assert info == {"n_frames": 40, "valid_ratio": 1.0, "hand_ratio": 1.0, "duration_ms": 39 * FPS_MS}


def test_presence_flags_and_missing_hand_is_zero():
    X, info = features(sequence((12, {"left_y": UP, "hands": ("left",)})))
    assert (X[:, LEFT_PRESENT] == 1).all() and (X[:, RIGHT_PRESENT] == 0).all()
    assert (X[:, 106:190] == 0).all()  # right hand block
    point9 = LEFT_LOCAL + 2 * 9
    np.testing.assert_allclose(X[:, point9:point9 + 2], [[0.0, -1.0]] * 32, atol=1e-5)  # straight up
    shoulder = 2 * POSE_IDX.index(11)
    np.testing.assert_allclose(X[:, shoulder:shoulder + 2], [[0.5, 0.0]] * 32, atol=1e-5)
    assert info["hand_ratio"] == 1.0


def test_no_hands_gives_hand_ratio_zero():
    X, info = features(sequence((8, {"hands": ()})))
    assert info["hand_ratio"] == 0.0 and info["valid_ratio"] == 1.0
    assert (X[:, 22:] == 0).all()


def test_invalid_frames_are_dropped_before_resampling():
    frames = sequence((10, {}))
    for f in frames[:5]:
        f["pose"] = None
    X, info = features(frames)
    assert info["valid_ratio"] == 0.5 and not np.isnan(X).any()
    assert (X[:, RIGHT_PRESENT] == 1).all()


def test_all_invalid_input_does_not_crash():
    X, info = features(sequence((6, {"shoulder_vis": 0.1})))
    assert X.shape == (32, 192) and X.dtype == np.float32 and not X.any()
    assert info["valid_ratio"] == 0.0 and info["hand_ratio"] == 0.0 and info["n_frames"] == 6


def test_empty_and_single_frame():
    X, info = features([])
    assert X.shape == (32, 192) and info["n_frames"] == 0 and info["valid_ratio"] == 0.0
    X, info = features([fake_frame(5.0)])
    assert np.allclose(X, X[0]) and info["duration_ms"] == 0.0


def test_custom_T():
    X, _ = features(sequence((10, {})), T=16)
    assert X.shape == (16, 192)
