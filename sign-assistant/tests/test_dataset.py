"""Tests for model/dataset.py (augmentation and the test-split guard).

Owner: A (Data/ML).
"""
import numpy as np
import pytest

from model.dataset import augment, load_split
from pose.features import arrays_to_features
from pose.normalize import frames_to_arrays
from tests.fake_pose import H, W, fake_frame, raise_both, sequence


def features(frames):
    return arrays_to_features(*frames_to_arrays(frames), W, H)[0]


def test_augment_shape_and_missing_hand_stays_zero():
    x = features(sequence((20, {"left_y": 0.6, "hands": ("left",)})))
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = augment(x, rng)
        assert a.shape == (32, 192) and a.dtype == np.float32 and np.isfinite(a).all()
        assert not a[:, 106:190].any() and not a[:, 191].any()  # right hand blocks and flag stay zero
        assert (a[:, 190] == 1).all()  # presence flags are not changed


def test_augment_changes_coordinates_but_not_sides():
    x = features(sequence((10, {}), (10, raise_both())))
    a = augment(x, np.random.default_rng(1))
    assert not np.allclose(a[:, :190], x[:, :190])
    # no mirroring: the signer's left shoulder (pose 11) keeps a positive body x
    assert (a[:, 10] > 0.3).all() and (a[:, 12] < -0.3).all()


def test_test_split_is_refused():
    with pytest.raises(ValueError):
        load_split("test")
