"""Tests for the row selection in pose/extract_dataset.py (no video or mediapipe needed).

Owner: A (Data/ML).
"""
from pose.extract_dataset import select_rows

VOCAB = [{"dataset_label": "MƏN"}, {"dataset_label": "SƏN"}, {"dataset_label": "GETMƏK"}]
ROWS = [{"video_id": f"{k:02d}", "dataset_label": label, "camera": camera}
        for k, (label, camera) in enumerate([("MƏN", "front"), ("SƏN", "side"), ("GETMƏK", "front"),
                                              ("YOX", "front"), ("MƏN", "unknown"), ("SƏN", "front")] * 3)]


def ids(rows):
    return [r["video_id"] for r in rows]


def test_vocab_and_camera_filter():
    rows = select_rows(ROWS, VOCAB)
    assert {r["dataset_label"] for r in rows} == {"MƏN", "SƏN", "GETMƏK"}
    assert all(r["camera"] == "front" for r in rows)
    assert len(select_rows(ROWS, VOCAB, camera="any")) == 15  # everything except YOX


def test_limit_classes_takes_vocab_order():
    rows = select_rows(ROWS, VOCAB, camera="any", limit_classes=2)
    assert {r["dataset_label"] for r in rows} == {"MƏN", "SƏN"}


def test_shards_split_the_work_without_overlap():
    full = ids(select_rows(ROWS, VOCAB, camera="any"))
    parts = [ids(select_rows(ROWS, VOCAB, camera="any", shard=(i, 3))) for i in (1, 2, 3)]
    assert sorted(sum(parts, [])) == sorted(full)
    assert all(len(p) == 5 for p in parts)
