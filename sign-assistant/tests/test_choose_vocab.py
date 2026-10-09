"""Tests for data/choose_vocab.py on a synthetic index.csv (CONTRACT v2: split by group).

Owner: A (Data/ML).
"""
import csv
import json
import re

import pytest

from data import choose_vocab as cv

HEADER = ["video_id", "video_path", "dataset_label", "signer_id", "camera", "n_frames", "fps", "group"]


def write_index(folder, rows):
    with open(folder / "index.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def make_rows(label, n_groups, per_group=4, camera="front", prefix="rec_"):
    return [[f"{cv.slug(label)}_{prefix}{g}_{k}", f"raw/{label}/{g}_{k}.mp4", label, "", camera, 20, 30,
             f"{prefix}{g:02d}"] for g in range(n_groups) for k in range(per_group)]


def run(folder, *args):
    cv.main(["--data-dir", str(folder), "--seeds", "20", *args])
    vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
    splits = json.loads((folder / "splits.json").read_text(encoding="utf-8"))
    return vocab, splits


def test_slug_is_ascii():
    assert cv.slug("BU GÜN") == "bu_gun"
    assert cv.slug("İSTƏMƏK") == "istemek"
    assert cv.slug("AĞRIMAQ") == "agrimaq"
    assert cv.slug("ÇƏKMƏK") == "cekmek"
    ids = [cv.slug(lab) for lab in cv.AZ]
    assert len(set(ids)) == len(ids) == 100
    assert all(re.fullmatch(r"[a-z0-9_]+", i) for i in ids)


def test_az_table_is_lowercase_azerbaijani():
    for lab, az in cv.AZ.items():
        assert az == az.strip() and "̇" not in az and not re.search(r"[A-ZƏİÖÜĞÇŞ]", az), lab
    assert cv.AZ["AĞRIMAQ"] == "ağrımaq" and cv.AZ["İNDİ"] == "indi"
    assert all(lab in cv.AZ for _, lab in cv.CANDIDATES if lab)


def test_end_to_end(tmp_path):
    rows = (make_rows("SABAH", 10) + make_rows("MƏN", 10, per_group=9) + make_rows("YOX", 10, per_group=6)
            + make_rows("SƏN", 5, per_group=8)  # 40 videos but only 5 groups -> dropped
            + make_rows("VAXT", 10, per_group=2)  # 20 videos -> dropped
            + make_rows("SƏNƏD", 10, camera="side")  # no front videos -> dropped
            + make_rows("QARABAQ", 10, per_group=3))
    write_index(tmp_path, rows)
    vocab, splits = run(tmp_path)
    # candidates first in priority order (MƏN before SABAH), then fill by video count
    assert [v["dataset_label"] for v in vocab] == ["MƏN", "SABAH", "YOX", "QARABAQ"]
    assert vocab[0] == {"id": "men", "gloss": "MƏN", "az": "mən", "dataset_label": "MƏN"}
    assert vocab[3]["gloss"] == "QARABAĞ" and vocab[3]["az"] == "qarabağ"

    assert set(splits) == {"seed", "train", "val", "test", "note"}
    assert not set(splits["train"]) & set(splits["val"])
    assert len(splits["val"]) == 2 and len(splits["train"]) == 8 and splits["test"] == []  # no team_* yet
    report = (tmp_path / "reports" / "vocab_choice.md").read_text(encoding="utf-8")
    assert "too few groups" in report and "not in dataset" in report and "EMPTY" in report


def test_team_groups_are_the_test_split(tmp_path):
    rows = make_rows("SABAH", 8) + make_rows("YOX", 8) + make_rows("SABAH", 2, prefix="team_")
    rows += [["nogroup_1", "raw/YOX/x.mp4", "YOX", "", "front", 20, 30, ""]]  # train-only, never listed
    write_index(tmp_path, rows)
    vocab, splits = run(tmp_path)
    assert splits["test"] == ["team_00", "team_01"]
    assert all(g.startswith("rec_") for g in splits["train"] + splits["val"])
    assert "" not in splits["train"] + splits["val"]


def test_every_class_is_in_val(tmp_path):
    rare = [r for r in make_rows("SU", 10) if r[7] == "rec_09"]  # SU exists only in group rec_09
    write_index(tmp_path, make_rows("SABAH", 10) + make_rows("YOX", 10) + rare)
    vocab, splits = run(tmp_path, "--min-signers", "1", "--min-videos", "1")
    assert "su" in [v["id"] for v in vocab]
    assert "rec_09" in splits["val"]


def test_max_classes_and_camera_any(tmp_path):
    write_index(tmp_path, make_rows("SABAH", 8) + make_rows("SƏNƏD", 8, camera="side") + make_rows("YOX", 8))
    vocab, _ = run(tmp_path, "--camera", "any", "--max-classes", "2")
    assert [v["id"] for v in vocab] == ["sabah", "sened"]


def test_empty_index_writes_nothing(tmp_path):
    write_index(tmp_path, [])
    with pytest.raises(SystemExit):
        cv.main(["--data-dir", str(tmp_path)])
    assert not (tmp_path / "vocab.json").exists()


def test_no_groups_fails(tmp_path):
    rows = make_rows("SABAH", 8)
    for r in rows:
        r[7] = ""
    write_index(tmp_path, rows)
    with pytest.raises(SystemExit):
        cv.main(["--data-dir", str(tmp_path), "--min-signers", "0"])
