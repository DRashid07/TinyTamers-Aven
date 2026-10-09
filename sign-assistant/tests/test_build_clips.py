"""Tests for the clip choice in data/build_clips.py (no video or ffmpeg needed).

Owner: D (Direction B/Speech/Eval).
"""
from data.build_clips import choose, cut_window, train_candidates

VOCAB = [{"id": "men", "dataset_label": "MƏN"}, {"id": "sen", "dataset_label": "SƏN"}]
SPLITS = {"train": ["rec_a", "rec_b"], "val": ["rec_v"], "test": ["team_x"]}


def row(vid, label="MƏN", group="rec_a", n_frames=15, camera="unknown"):
    return {"video_id": vid, "dataset_label": label, "group": group, "camera": camera, "n_frames": str(n_frames),
            "fps": "30.0", "signer_id": "", "video_path": f"raw/{vid}.mp4"}


def info(segment=True, hand=1.0, valid=1.0, start=100.0, end=400.0):
    return {"segment": segment, "hand_ratio": hand, "valid_ratio": valid, "start_ms": start, "end_ms": end}


def test_only_train_groups_and_camera():
    rows = [row("a"), row("b", group="rec_v"), row("c", group="team_x"), row("d", group=""),
            row("e", label="SƏN", group="rec_b"), row("f", camera="side")]
    cands = train_candidates(rows, VOCAB, SPLITS, "any")
    assert [r["video_id"] for r in cands["men"]] == ["a", "d", "f"]  # empty group = train-only
    assert [r["video_id"] for r in cands["sen"]] == ["e"]
    assert [r["video_id"] for r in train_candidates(rows, VOCAB, SPLITS, "side")["men"]] == ["f"]


def test_prefers_a_found_segment_then_hand_ratio():
    rows = [row("a"), row("b"), row("c")]
    picked, _, checked = choose(rows, [info(segment=False), info(hand=0.6), info(hand=0.9)])
    assert picked["video_id"] == "c" and checked


def test_without_landmarks_takes_the_median_clip_unchecked():
    rows = [row("a", n_frames=10), row("b", n_frames=20), row("c", n_frames=40)]
    picked, picked_info, checked = choose(rows, [None, None, None])
    assert picked["video_id"] == "b" and picked_info is None and not checked


def test_cut_window_pads_300_ms_inside_the_clip():
    r = row("a", n_frames=30)  # 1000 ms
    assert cut_window(r, info(start=500.0, end=600.0)) == (200.0, 900.0)
    assert cut_window(r, info(start=100.0, end=900.0)) == (0.0, 1000.0)  # clamped to the clip
    assert cut_window(r, None) == (0.0, 1000.0)  # whole clip without landmarks
