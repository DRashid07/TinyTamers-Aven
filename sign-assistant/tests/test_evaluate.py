"""Tests for eval/evaluate.py: metrics, the test-run log rule, the team path and the report sections.

Owner: D (Direction B/Speech/Eval).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from eval import evaluate as ev
from tests.fake_pose import H, W, raise_both, sequence

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "model" / "artifacts"
VOCAB = [{"id": "a", "gloss": "A"}, {"id": "b", "gloss": "B"}, {"id": "c", "gloss": "C"}]
CFG = {"classes": ["a", "b", "c"], "tau": 0.6, "margin": 0.1, "min_valid_ratio": 0.5, "min_hand_ratio": 0.5}
GOOD = {"valid_ratio": 1.0, "hand_ratio": 1.0}


def test_metrics():
    probs = np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.4, 0.35, 0.25], [0.0, 0.1, 0.9]])
    y = np.array([0, 0, 1, 1, 2])
    m = ev.metrics(probs, y, [GOOD] * 5, CFG, VOCAB)
    assert m["top1"] == pytest.approx(3 / 5)
    assert m["macro"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert m["pairs"][0][1:] == (0, 1) and m["pairs"][0][0] == pytest.approx(0.5)  # (0.5 + 0.5) / 2
    coverage, selective, answered, right = m["chosen"]  # row 3 (p1 0.4) abstains, row 1 is a wrong answer
    assert (answered, right) == (4, 3) and coverage == pytest.approx(0.8) and selective == pytest.approx(0.75)
    assert [t for t, *_ in m["taus"]] == ev.TAUS


def test_test_log_rule(tmp_path):
    log = tmp_path / "test_runs.log"
    ev.check_test_log(log, "abc123", "")  # no log yet: allowed
    log.write_text("2026-10-09 18:15 UTC\tcommit=x\tconfig_sha256=abc123\tclips=5\treason=first run\n")
    with pytest.raises(SystemExit):
        ev.check_test_log(log, "abc123", "")
    ev.check_test_log(log, "abc123", "fixed a bug in the features")  # with a reason: allowed
    ev.check_test_log(log, "other", "")  # another config: allowed


def test_test_split_needs_final():
    with pytest.raises(SystemExit):
        ev.main(["--split", "test"])


def write_records(folder):
    folder.mkdir()
    sign = sequence((5, {}), (15, raise_both()), (5, {}))
    still = sequence((20, {}))  # hands never raised: segment_offline finds nothing, the whole clip is used
    for name, rec_id, frames in [("s1_men_1", "men", sign), ("s2_sen_1", "sen", still), ("s1_x_1", "nope", sign)]:
        (folder / f"{name}.json").write_text(json.dumps({"id": rec_id, "signer": name[:2], "w": W, "h": H,
                                                         "frames": frames}), encoding="utf-8")


def test_team_split(tmp_path):
    write_records(tmp_path / "team")
    vocab = [{"id": "men", "gloss": "MƏN"}, {"id": "sen", "gloss": "SƏN"}]
    data = ev.team_split(tmp_path / "team", vocab, 32)
    assert data["X"].shape == (2, 32, 192) and data["y"].tolist() == [0, 1]
    assert data["units"] == ["s1", "s2"] and data["no_segment"] == 1 and data["missing"] == 1


def test_team_split_reads_signer_subfolders(tmp_path):
    # data/team_recordings/README.md layout: <signer>/<id>_<n>.json
    for signer in ("aysel", "rashid"):
        folder = tmp_path / "team" / signer
        folder.mkdir(parents=True)
        (folder / "men_1.json").write_text(json.dumps({"id": "men", "signer": signer, "w": W, "h": H,
                                                       "frames": sequence((5, {}), (15, raise_both()), (5, {}))}))
    (tmp_path / "team" / "README.md").write_text("not a recording")
    data = ev.team_split(tmp_path / "team", [{"id": "men", "gloss": "MƏN"}], 32)
    assert data["n"] == 2 and data["units"] == ["aysel", "rashid"] and data["no_segment"] == 0


def test_report_sections_keep_their_order(tmp_path):
    report = tmp_path / "REPORT.md"
    ev.write_section(report, "team", "<!-- BEGIN team -->\nT1\n<!-- END team -->")
    ev.write_section(report, "val", "<!-- BEGIN val -->\nV1\n<!-- END val -->")
    ev.write_section(report, "team", "<!-- BEGIN team -->\nT2\n<!-- END team -->")
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Evaluation report") and text.index("V1") < text.index("T2") and "T1" not in text


@pytest.mark.skipif(not (ARTIFACTS / "model.pt").exists(), reason="no trained model in model/artifacts")
def test_team_and_test_runs_end_to_end(tmp_path, monkeypatch):
    write_records(tmp_path / "team")
    out = tmp_path / "eval"
    ev.main(["--split", "team", "--team-dir", str(tmp_path / "team"), "--eval-dir", str(out)])
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "Team recordings (non-native signers, webcam)" in report and (out / "figures" / "confusion_team.png").exists()

    # the test protocol, with the team records standing in for the (still empty) test groups
    monkeypatch.setattr(ev, "dataset_split", lambda split, vocab, data_dir, T: ev.team_split(tmp_path / "team", vocab, T))
    ev.main(["--split", "test", "--final", "--eval-dir", str(out)])
    assert (out / "test_runs.log").read_text(encoding="utf-8").count("reason=first run") == 1
    with pytest.raises(SystemExit):
        ev.main(["--split", "test", "--final", "--eval-dir", str(out)])
    ev.main(["--split", "test", "--final", "--eval-dir", str(out), "--rerun-reason", "test of the log"])
    log = (out / "test_runs.log").read_text(encoding="utf-8").splitlines()
    assert len(log) == 2 and "reason=test of the log" in log[1]
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert report.index("BEGIN test") < report.index("BEGIN team")
