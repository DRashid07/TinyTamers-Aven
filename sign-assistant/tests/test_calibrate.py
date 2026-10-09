"""Tests for model/calibrate.py with synthetic logits and a fake model (no trained model needed).

Owner: A (Data/ML).
"""
import csv
import json

import numpy as np
import pytest
import torch

from model import calibrate

VOCAB = [{"id": i, "gloss": i.upper(), "az": i, "dataset_label": i.upper()} for i in ("a", "b", "c")]
CFG = {"feature_version": 1, "T": 32, "n_features": 192, "arch": "gru", "classes": ["a", "b", "c"],
       "temperature": 1.0, "tau": 0.7, "margin": 0.15, "min_valid_ratio": 0.6, "min_hand_ratio": 0.5}
GOOD = {"valid_ratio": 1.0, "hand_ratio": 1.0}


def test_fit_temperature_undoes_overconfidence():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4000, 5)) * 2
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(5, p=p) for p in probs])
    temperature = calibrate.fit_temperature(logits * 3, labels)  # a model 3x too confident
    assert 2.6 < temperature < 3.4
    assert calibrate.nll(logits * 3, labels, temperature) < calibrate.nll(logits * 3, labels)


def test_ece():
    labels = np.array([0, 0, 1, 1])
    assert calibrate.ece(np.array([[1.0, 0.0]] * 4), labels) == pytest.approx(0.5)  # sure, right half the time
    assert calibrate.ece(np.array([[0.75, 0.25]] * 4), np.array([0, 0, 0, 1])) == pytest.approx(0.0)


def test_sweep_uses_decide_and_choose_picks_highest_coverage():
    probs = np.array([[0.9, 0.05, 0.05],    # right, confident
                      [0.6, 0.35, 0.05],    # right, margin 0.25
                      [0.5, 0.45, 0.05],    # wrong (label b), margin 0.05
                      [0.95, 0.03, 0.02]])  # right, but no hands: decide always abstains
    labels = np.array([0, 0, 1, 0])
    infos = [GOOD, GOOD, GOOD, {"valid_ratio": 1.0, "hand_ratio": 0.0}]
    rows = calibrate.sweep(probs, labels, infos, CFG, VOCAB, taus=[0.4, 0.55, 0.8], margins=[0.0, 0.2])
    by = {(r["tau"], r["margin"]): r for r in rows}

    assert (by[0.4, 0.0]["accepted"], by[0.4, 0.0]["correct"], by[0.4, 0.0]["coverage"]) == (3, 2, 0.75)
    assert (by[0.4, 0.2]["coverage"], by[0.4, 0.2]["selective_accuracy"]) == (0.5, 1.0)
    assert (by[0.8, 0.0]["coverage"], by[0.8, 0.0]["selective_accuracy"]) == (0.25, 1.0)

    chosen, reached = calibrate.choose(rows, target=0.9)
    assert reached and (chosen["tau"], chosen["margin"], chosen["coverage"]) == (0.55, 0.2, 0.5)  # ties: stricter

    chosen, reached = calibrate.choose([by[0.4, 0.0]], target=0.9)
    assert not reached and chosen is by[0.4, 0.0]


def test_update_config_keeps_other_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(CFG), encoding="utf-8")
    calibrate.update_config(path, 1.23456, 0.6, 0.1)
    assert json.loads(path.read_text(encoding="utf-8")) == {**CFG, "temperature": 1.2346, "tau": 0.6, "margin": 0.1}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = torch.nn.Linear(192, 3)

    def forward(self, x):
        return self.linear(x.mean(dim=1)) * 5


class FakePredictor:
    """Stands in for model.predictor.Predictor until model/net.py exists."""

    def __init__(self, model_dir):
        self.loaded, self.model = True, FakeModel()
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))


def fake_landmarks(path, rng, n=20):
    pose = np.concatenate([rng.uniform(0.3, 0.7, (n, 33, 3)), np.ones((n, 33, 1))], axis=2)
    np.savez(path, pose=pose.astype(np.float32), hands=rng.uniform(0.2, 0.8, (n, 2, 21, 3)).astype(np.float32),
             t=np.arange(n) * 33.3, w=1280, h=960)


def test_main_reads_only_val_and_writes_artifacts(tmp_path, monkeypatch, capsys):
    data, model_dir = tmp_path / "data", tmp_path / "artifacts"
    (data / "landmarks").mkdir(parents=True)
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(CFG), encoding="utf-8")
    (data / "vocab.json").write_text(json.dumps(VOCAB), encoding="utf-8")
    (data / "splits.json").write_text(json.dumps({"seed": 1, "train": [], "val": ["rec_v"], "test": ["team_x"]}),
                                      encoding="utf-8")
    rng = np.random.default_rng(1)
    rows = [{"video_id": f"v{k}", "dataset_label": "ABC"[k % 3], "group": "rec_v"} for k in range(12)]
    for row in rows:
        fake_landmarks(data / "landmarks" / f"{row['video_id']}.npz", rng)
    rows.append({"video_id": "testclip", "dataset_label": "A", "group": "team_x"})
    (data / "landmarks" / "testclip.npz").write_bytes(b"not an npz: reading the test split would fail")
    with open(data / "index.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "dataset_label", "group"])
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(calibrate, "Predictor", FakePredictor)

    calibrate.main(["--model-dir", str(model_dir), "--data-dir", str(data), "--target", "0.0"])

    out = capsys.readouterr().out
    assert "VAL only: 12 clips" in out and "Chosen" in out
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert config["tau"] in calibrate.TAUS and config["margin"] in calibrate.MARGINS and config["temperature"] > 0
    assert {k: v for k, v in config.items() if k not in ("temperature", "tau", "margin")} == \
        {k: v for k, v in CFG.items() if k not in ("temperature", "tau", "margin")}
    with open(model_dir / "val_threshold_sweep.csv", encoding="utf-8", newline="") as f:
        sweep = list(csv.DictReader(f))
    assert len(sweep) == len(calibrate.TAUS) * len(calibrate.MARGINS)
    assert list(sweep[0]) == calibrate.SWEEP_COLUMNS
