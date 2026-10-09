"""Tests for the clip choice in data/build_clips.py (no video or ffmpeg needed).

Owner: D (Direction B/Speech/Eval).
"""
import csv
import json
from pathlib import Path

import pytest

from data import build_clips
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


def write_data(data_dir, vocab, rows, index=None):
    """Minimal independent dataset for exercising the CLI without source videos."""
    (data_dir / "clips").mkdir()
    (data_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (data_dir / "splits.json").write_text(json.dumps(SPLITS), encoding="utf-8")
    with (data_dir / "index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if index is not None:
        (data_dir / "clips_index.json").write_text(json.dumps(index), encoding="utf-8")


def test_playback_vocabulary_covers_every_dataset_label_and_keeps_recognition_ids():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    recognition = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))
    playback = json.loads((data_dir / "playback_vocab.json").read_text(encoding="utf-8"))
    with (data_dir / "index.csv").open(encoding="utf-8", newline="") as f:
        labels = {r["dataset_label"] for r in csv.DictReader(f)}
    assert len(recognition) <= len(playback)
    playback_by_id = {entry["id"]: entry for entry in playback}
    assert all(playback_by_id[entry["id"]] == entry for entry in recognition)
    assert {v["dataset_label"] for v in playback} == labels
    assert len(playback) == len({v["id"] for v in playback}) == 100
    by_id = {v["id"]: v for v in playback}
    assert by_id["salam"]["dataset_label"] == "SALAM"
    assert by_id["nece"]["dataset_label"] == "NECƏ"
    assert by_id["yaxsi"]["dataset_label"] == "YAXŞI"


def test_separate_vocab_skip_existing_preserves_video_and_source_metadata(tmp_path, monkeypatch):
    vocab = [{**v, "gloss": v["dataset_label"]} for v in VOCAB]
    original = {"source_video_id": "original", "checked": True, "note": "reviewed clip"}
    write_data(tmp_path, vocab[:1], [row("new", label="SƏN"), row("val", label="SƏN", group="rec_v")],
               {"men": original})
    (tmp_path / "clips" / "men.mp4").write_bytes(b"original video")
    playback_path = tmp_path / "playback_vocab.json"
    playback_path.write_text(json.dumps(vocab), encoding="utf-8")
    calls = []

    def transcode(ffmpeg, src, out, start_ms, end_ms):
        calls.append((src, out))
        out.write_bytes(b"new video")
        return True, ""

    monkeypatch.setattr(build_clips, "transcode", transcode)
    monkeypatch.setattr(build_clips, "landmark_info", lambda *args: None)
    build_clips.main(["--camera", "any", "--data-dir", str(tmp_path), "--vocab-file", str(playback_path),
                      "--skip-existing", "--ffmpeg", "fake-ffmpeg"])

    index = json.loads((tmp_path / "clips_index.json").read_text(encoding="utf-8"))
    assert index["men"] == original
    assert (tmp_path / "clips" / "men.mp4").read_bytes() == b"original video"
    assert index["sen"]["source_video_id"] == "new"
    assert index["sen"]["checked"] is False
    assert len(calls) == 1 and calls[0][0].name == "new.mp4"
    assert json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8")) == vocab[:1]


@pytest.mark.parametrize("case", ["empty_clip", "missing_clip", "missing_entry"])
def test_skip_existing_rebuilds_missing_or_unindexed_clip(tmp_path, monkeypatch, case):
    vocab = [{**VOCAB[0], "gloss": "MƏN"}]
    index = {} if case == "missing_entry" else {"men": {"source_video_id": "old"}}
    write_data(tmp_path, vocab, [row("new")], index)
    if case != "missing_clip":
        (tmp_path / "clips" / "men.mp4").write_bytes(b"" if case == "empty_clip" else b"unindexed")

    def transcode(ffmpeg, src, out, start_ms, end_ms):
        out.write_bytes(b"rebuilt")
        return True, ""

    monkeypatch.setattr(build_clips, "transcode", transcode)
    monkeypatch.setattr(build_clips, "landmark_info", lambda *args: None)
    build_clips.main(["--camera", "any", "--data-dir", str(tmp_path), "--skip-existing",
                      "--ffmpeg", "fake-ffmpeg"])
    saved = json.loads((tmp_path / "clips_index.json").read_text(encoding="utf-8"))
    assert saved["men"]["source_video_id"] == "new"
    assert (tmp_path / "clips" / "men.mp4").read_bytes() == b"rebuilt"


def test_skip_existing_complete_library_needs_no_ffmpeg(tmp_path, monkeypatch):
    vocab = [{**VOCAB[0], "gloss": "MƏN"}]
    index = {"men": {"source_video_id": "original", "checked": True}}
    write_data(tmp_path, vocab, [row("new")], index)
    (tmp_path / "clips" / "men.mp4").write_bytes(b"original video")
    monkeypatch.setattr(build_clips.shutil, "which", lambda *args: None)
    monkeypatch.setattr(build_clips, "transcode", lambda *args: pytest.fail("existing clip must be kept"))
    build_clips.main(["--camera", "any", "--data-dir", str(tmp_path), "--skip-existing"])
    assert json.loads((tmp_path / "clips_index.json").read_text(encoding="utf-8")) == index
