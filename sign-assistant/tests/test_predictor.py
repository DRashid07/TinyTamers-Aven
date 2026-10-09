"""Tests for model/predictor.py with a tiny stand-in network (model/net.py belongs to A),
plus one test of the real artifacts in model/artifacts when they exist (they are not in git).

Owner: C (Backend/LLM).
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from model import net
from model.predictor import Predictor
from pose.features import arrays_to_features
from pose.normalize import frames_to_arrays
from tests.fake_pose import H, W, raise_both, sequence

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "model" / "artifacts"

CONFIG = {"feature_version": 1, "T": 32, "n_features": 192, "arch": "gru", "classes": ["men", "sen", "hekim"],
          "temperature": 2.0, "tau": 0.7, "margin": 0.15, "min_valid_ratio": 0.6, "min_hand_ratio": 0.5}


class Tiny(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.fc = torch.nn.Linear(192, n_classes)

    def forward(self, x):
        return self.fc(x.mean(dim=1))


def save(folder, config, model):
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.save(model.state_dict(), folder / "model.pt")


def test_missing_artifacts_leave_it_unloaded(tmp_path):
    assert not Predictor(tmp_path).loaded


def test_loads_and_applies_temperature(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = Tiny(3)
    save(tmp_path, CONFIG, model)
    monkeypatch.setattr(net, "build_model", lambda cfg: Tiny(len(cfg["classes"])))
    predictor = Predictor(tmp_path)
    assert predictor.loaded
    X = np.random.default_rng(0).normal(size=(32, 192)).astype(np.float32)
    probs = predictor.probs(X)
    assert probs.shape == (3,) and abs(probs.sum() - 1) < 1e-5
    with torch.no_grad():
        expected = torch.softmax(model(torch.as_tensor(X)[None])[0] / 2.0, dim=0).numpy()
    np.testing.assert_allclose(probs, expected, rtol=1e-5)


def test_wrong_feature_version_is_refused(tmp_path, monkeypatch):
    save(tmp_path, {**CONFIG, "feature_version": 2}, Tiny(3))
    monkeypatch.setattr(net, "build_model", lambda cfg: Tiny(3))
    assert not Predictor(tmp_path).loaded


def test_weights_that_do_not_fit_the_net_are_not_loaded(tmp_path, monkeypatch):
    save(tmp_path, CONFIG, Tiny(3))
    monkeypatch.setattr(net, "build_model", lambda cfg: Tiny(5))
    assert not Predictor(tmp_path).loaded


@pytest.mark.parametrize("change", [
    {"n_features": 190}, {"temperature": 0.0}, {"tau": 0.0}, {"tau": 1.5}, {"margin": -0.1}, {"classes": []},
    {"tau": None},  # None = key removed
])
def test_bad_config_is_refused(tmp_path, monkeypatch, change):
    config = {k: v for k, v in {**CONFIG, **change}.items() if v is not None}
    save(tmp_path, config, Tiny(3))
    monkeypatch.setattr(net, "build_model", lambda cfg: Tiny(3))
    assert not Predictor(tmp_path).loaded


def test_config_saved_with_a_bom_still_loads(tmp_path, monkeypatch):
    save(tmp_path, CONFIG, Tiny(3))
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8-sig")  # e.g. PowerShell 5.1
    monkeypatch.setattr(net, "build_model", lambda cfg: Tiny(3))
    assert Predictor(tmp_path).loaded


def test_load_logs_the_warm_up_time(tmp_path, monkeypatch, caplog):
    save(tmp_path, CONFIG, Tiny(3))
    monkeypatch.setattr(net, "build_model", lambda cfg: Tiny(3))
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        assert Predictor(tmp_path).loaded
    assert "warm-up inference" in caplog.text and "tau 0.7" in caplog.text


@pytest.mark.skipif(not (REAL / "model.pt").exists(), reason="no trained model in model/artifacts")
def test_real_model_loads_and_is_fast_enough():
    predictor = Predictor(REAL)
    assert predictor.loaded
    vocab = json.loads((ROOT / "data" / "vocab.json").read_text(encoding="utf-8"))
    assert predictor.config["classes"] == [v["id"] for v in vocab]
    frames = sequence((30, raise_both()))  # one second of signing at 30 fps
    times = []
    for _ in range(20):
        start = time.perf_counter()
        pose, hands, t = frames_to_arrays(frames)
        X, _ = arrays_to_features(pose, hands, t, W, H, T=predictor.config["T"])
        predictor.probs(X)
        times.append((time.perf_counter() - start) * 1000)
    print(f"features + inference per segment: median {np.median(times):.1f} ms, max {max(times):.1f} ms")
    assert np.median(times) < 50
