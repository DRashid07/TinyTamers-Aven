"""Tests for model/predictor.py with a tiny stand-in network (model/net.py belongs to A).

Owner: C (Backend/LLM).
"""
import json

import numpy as np
import torch

from model import net
from model.predictor import Predictor

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
