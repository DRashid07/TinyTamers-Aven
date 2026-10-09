"""Tests for pose/segment.py on synthetic frames (30 fps, default segment_config.json).

Owner: C (Backend/LLM).
"""
import pytest

from pose.normalize import frames_to_arrays
from pose.segment import Segmenter, load_config, segment_offline
from tests.fake_pose import DOWN, H, W, raise_both, sequence

CFG = load_config()
DOWN_10 = (10, {})
SIGN_15 = (15, raise_both())  # 14 * 33.3 ms = 467 ms


def events(frames):
    seg = Segmenter(W, H, CFG)
    return [e for e in (seg.push(f) for f in frames) if e is not None]


def offline(frames):
    return segment_offline(*frames_to_arrays(frames), W, H, CFG)


def test_config_defaults():
    assert CFG == {"raise_y": 1.0, "start_frames": 3, "end_frames": 6, "min_ms": 300, "max_ms": 4000}


def test_down_up_down_gives_one_ok_segment():
    frames = sequence(DOWN_10, SIGN_15, DOWN_10)
    ev = events(frames)
    assert [e["event"] for e in ev] == ["start", "end"]
    assert ev[1]["status"] == "ok"
    assert ev[1]["frames"] == frames[10:25]  # first to last raised frame


def test_start_needs_three_up_frames_and_end_needs_six_down():
    seg = Segmenter(W, H, CFG)
    frames = sequence(DOWN_10, SIGN_15, DOWN_10)
    first_event = {i: e["event"] for i, e in enumerate(seg.push(f) for f in frames) if e}
    assert first_event == {12: "start", 30: "end"}


@pytest.mark.parametrize("n_up", [1, 2])
def test_flicker_gives_no_segment(n_up):
    frames = sequence(DOWN_10, (n_up, raise_both()), DOWN_10)
    assert events(frames) == [] and offline(frames) is None


def test_short_dip_does_not_split_a_sign():
    frames = sequence(DOWN_10, (8, raise_both()), (3, {}), (8, raise_both()), DOWN_10)
    assert [e["event"] for e in events(frames)] == ["start", "end"]


def test_very_short_raise_is_too_short():
    ev = events(sequence(DOWN_10, (5, raise_both()), DOWN_10))  # 4 * 33.3 = 133 ms
    assert ev[-1]["event"] == "end" and ev[-1]["status"] == "too_short"


def test_very_long_raise_is_too_long():
    ev = events(sequence(DOWN_10, (150, raise_both()), DOWN_10))  # about 5 s
    assert ev[-1]["status"] == "too_long"


def test_one_hand_is_enough():
    ev = events(sequence(DOWN_10, (15, {"left_y": 0.6, "right_y": DOWN}), DOWN_10))
    assert ev[-1]["status"] == "ok"


def test_two_separated_signs_give_two_segments():
    frames = sequence(DOWN_10, SIGN_15, DOWN_10, SIGN_15, DOWN_10)
    ends = [e for e in events(frames) if e["event"] == "end"]
    assert len(ends) == 2 and all(e["status"] == "ok" for e in ends)
    assert ends[0]["frames"] == frames[10:25] and ends[1]["frames"] == frames[35:50]


def test_segment_offline_first_start_to_last_end():
    frames = sequence(DOWN_10, SIGN_15, DOWN_10, SIGN_15, DOWN_10)
    assert offline(frames) == (10, 50)  # end is exclusive: frames[10:50]
    assert offline(sequence(DOWN_10, SIGN_15, DOWN_10)) == (10, 25)


def test_segment_offline_matches_live_and_closes_an_open_sign():
    frames = sequence(DOWN_10, SIGN_15, DOWN_10)
    start, end = offline(frames)
    assert events(frames)[-1]["frames"] == frames[start:end]
    assert offline(sequence(DOWN_10, SIGN_15)) == (10, 25)  # recording stopped with hands up


def test_no_pose_or_no_hands_never_starts():
    frames = sequence((20, {"pose": False, **raise_both()}), (20, {"hands": ()}))
    assert events(frames) == [] and offline(frames) is None
