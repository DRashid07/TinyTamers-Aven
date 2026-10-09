"""Tests for data/choose_vocab.py on a synthetic index.csv.

Owner: A (Data/ML).
"""
import csv
import json
import re

import pytest

from data import choose_vocab as cv

HEADER = ["video_id", "video_path", "dataset_label", "signer_id", "camera", "n_frames", "fps"]


def write_index(folder, rows):
    with open(folder / "index.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def make_rows(label, n_signers, per_signer=4, camera="front"):
    return [[f"{cv.slug(label)}_{s}_{k}", f"raw/{label}/{s}_{k}.mp4", label, f"s{s:02d}", camera, 20, 30]
            for s in range(n_signers) for k in range(per_signer)]


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
    rows = (make_rows("SABAH", 10) + make_rows("MƏN", 10, per_signer=9) + make_rows("YOX", 10, per_signer=6)
            + make_rows("SƏN", 5, per_signer=8)  # 40 videos but only 5 signers -> dropped
            + make_rows("VAXT", 10, per_signer=2)  # 20 videos -> dropped
            + make_rows("SƏNƏD", 10, camera="side")  # no front videos -> dropped
            + make_rows("QARABAQ", 10, per_signer=3))
    write_index(tmp_path, rows)
    cv.main(["--data-dir", str(tmp_path), "--seeds", "20"])

    vocab = json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8"))
    # candidates first in priority order (MƏN before SABAH), then fill by video count
    assert [v["dataset_label"] for v in vocab] == ["MƏN", "SABAH", "YOX", "QARABAQ"]
    assert vocab[0] == {"id": "men", "gloss": "MƏN", "az": "mən", "dataset_label": "MƏN"}
    assert vocab[3]["gloss"] == "QARABAĞ" and vocab[3]["az"] == "qarabağ"

    splits = json.loads((tmp_path / "splits.json").read_text(encoding="utf-8"))
    assert set(splits) == {"seed", "train", "val", "test", "note"}
    sets = [set(splits[n]) for n in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(sets[1]) == len(sets[2]) == 2 and len(sets[0]) == 6
    report = (tmp_path / "reports" / "vocab_choice.md").read_text(encoding="utf-8")
    assert "too few signers" in report and "not in dataset" in report


def test_max_classes_and_camera_any(tmp_path):
    write_index(tmp_path, make_rows("SABAH", 8) + make_rows("SƏNƏD", 8, camera="side") + make_rows("YOX", 8))
    cv.main(["--data-dir", str(tmp_path), "--camera", "any", "--max-classes", "2", "--seeds", "5"])
    vocab = json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8"))
    assert [v["id"] for v in vocab] == ["sabah", "sened"]


def test_empty_index_writes_nothing(tmp_path):
    write_index(tmp_path, [])
    with pytest.raises(SystemExit):
        cv.main(["--data-dir", str(tmp_path)])
    assert not (tmp_path / "vocab.json").exists()


def test_no_signer_ids_fails(tmp_path):
    rows = make_rows("SABAH", 8)
    for r in rows:
        r[3] = ""
    write_index(tmp_path, rows)
    with pytest.raises(SystemExit):
        cv.main(["--data-dir", str(tmp_path), "--min-signers", "0"])
