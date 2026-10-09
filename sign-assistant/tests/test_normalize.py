"""Tests for pose/normalize.py on synthetic frames.

Owner: C (Backend/LLM).
"""
import numpy as np
import pytest

from pose.normalize import assign_hands, frames_to_arrays, to_body
from tests.fake_pose import DOWN, H, UP, W, fake_frame, sequence


def body_of(frames, w=W, h=H):
    pose, hands, _ = frames_to_arrays(frames)
    return to_body(pose, pose, w, h)


def test_frames_to_arrays_shapes_and_missing():
    header = {"type": "start", "w": W, "h": H}
    frames = [header, fake_frame(0.0), fake_frame(33.3, hands=("left",), pose=False)]
    pose, hands, t = frames_to_arrays(frames)
    assert pose.shape == (2, 33, 4) and pose.dtype == np.float32
    assert hands.shape == (2, 2, 21, 3) and hands.dtype == np.float32
    assert t.tolist() == [0.0, 33.3]
    assert np.isnan(pose[1]).all() and np.isnan(hands[1, 1]).all() and not np.isnan(hands[1, 0]).any()


@pytest.mark.parametrize("shift, zoom", [((0.1, -0.05), 1.0), ((0.0, 0.0), 0.6), ((-0.03, 0.02), 1.3)])
def test_translation_and_scale_do_not_change_body_coords(shift, zoom):
    ref, _ = body_of([fake_frame(0.0)])
    moved, valid = body_of([fake_frame(0.0, shift=shift, zoom=zoom)])
    assert valid.all()
    np.testing.assert_allclose(moved, ref, atol=1e-5)


def test_aspect_fix_is_applied():
    body, _ = body_of([fake_frame(0.0)])
    np.testing.assert_allclose(body[0, 11], [0.5, 0.0], atol=1e-6)  # signer's left shoulder, image right
    np.testing.assert_allclose(body[0, 0], [0.0, -0.2 / (0.2 * W / H)], atol=1e-6)  # nose, y points down
    square, _ = body_of([fake_frame(0.0)], w=480, h=480)  # without the 4:3 stretch the nose is higher
    np.testing.assert_allclose(square[0, 0], [0.0, -1.0], atol=1e-6)


@pytest.mark.parametrize("frame", [fake_frame(0.0, shoulder_vis=0.3), fake_frame(0.0, pose=False)])
def test_low_visibility_or_missing_shoulder_is_invalid(frame):
    pose, _, _ = frames_to_arrays([frame])
    body, valid = to_body(pose, pose, W, H)
    assert not valid[0] and np.isnan(body[0]).all()


def test_nan_shoulder_is_invalid_and_valid_frames_have_no_nan():
    pose, hands, _ = frames_to_arrays(sequence((3, {})))
    pose[1, 11] = np.nan
    body, valid = to_body(pose, pose, W, H)
    assert valid.tolist() == [True, False, True]
    assert not np.isnan(body[valid]).any()
    hand_body, _ = to_body(pose, hands[:, 0], W, H)
    assert not np.isnan(hand_body[valid]).any()


@pytest.mark.parametrize("order", [("left", "right"), ("right", "left")])
def test_assign_hands_follows_the_nearer_wrist(order):
    pose, hands, _ = frames_to_arrays([fake_frame(0.0, left_y=UP, right_y=DOWN, hands=order)])
    left, right = assign_hands(pose, hands)
    assert left[0, 0, 1] == pytest.approx(UP) and right[0, 0, 1] == pytest.approx(DOWN)
    assert left[0, 0, 0] > right[0, 0, 0]  # signer's left hand is on the image right


def test_assign_single_hand_in_any_slot():
    pose, hands, _ = frames_to_arrays([fake_frame(0.0, hands=("left",))])
    hands = hands[:, ::-1]  # the detected hand now sits in slot 1
    left, right = assign_hands(pose, hands)
    assert not np.isnan(left).any() and np.isnan(right).all()


def test_assign_without_pose_drops_hands():
    pose, hands, _ = frames_to_arrays([fake_frame(0.0, pose=False)])
    left, right = assign_hands(pose, hands)
    assert np.isnan(left).all() and np.isnan(right).all()
