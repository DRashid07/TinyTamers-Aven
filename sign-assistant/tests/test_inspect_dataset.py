"""Tests for data/inspect_dataset.py on a tiny fake dataset (no network, no real data).

Owner: A (Data/ML).
"""
import csv
import json

import cv2
import numpy as np

from data.inspect_dataset import COLUMNS, build_index, load_recording_dates, summarise, write_index

KEY_A, KEY_B = "a" * 32, "b" * 32


def write_video(path, n_frames, fps=30):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(n_frames):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()


def test_index_rows_groups_and_csv(tmp_path):
    words = tmp_path / "data" / "raw" / "AzSLD_Words_100"
    write_video(words / "MƏN" / f"{KEY_A}.mp4", 15)
    write_video(words / "SİZ" / f"{KEY_B}.mp4", 30)
    ann = tmp_path / "ann" / "AzSLD_Sentences" / "7" / "ann"
    ann.mkdir(parents=True)
    (ann / "2022-05-31 15-58-30.mp4.json").write_text(
        json.dumps({"tags": [{"name": "MƏN", "key": KEY_A}]}), encoding="utf-8")

    rows, unreadable, resolutions = build_index(words, load_recording_dates(tmp_path / "ann"), root=tmp_path)

    assert unreadable == [] and resolutions == {"64x48": 2}
    men, siz = rows
    assert men["video_id"] == KEY_A
    assert men["video_path"] == f"data/raw/AzSLD_Words_100/MƏN/{KEY_A}.mp4"
    assert (men["dataset_label"], men["n_frames"], men["fps"], men["group"]) == ("MƏN", 15, 30.0, "rec_2022-05-31")
    assert (siz["dataset_label"], siz["signer_id"], siz["camera"], siz["group"]) == ("SİZ", "", "unknown", "")

    out = tmp_path / "index.csv"
    write_index(rows, out)
    with open(out, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        assert [r["dataset_label"] for r in reader] == ["MƏN", "SİZ"]

    summary = summarise(rows, unreadable, resolutions)
    assert "| classes | 2 |" in summary and "| videos without a recording date | 1 |" in summary
