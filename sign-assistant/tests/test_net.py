"""Tests for model/net.py.

Owner: A (Data/ML).
"""
import pytest
import torch

from model.net import build_model

CONFIG = {"feature_version": 1, "T": 32, "n_features": 192, "classes": ["a", "b", "c"]}


@pytest.mark.parametrize("arch", ["gru", "transformer"])
def test_forward_shape(arch):
    model = build_model({**CONFIG, "arch": arch}).eval()
    with torch.no_grad():
        logits = model(torch.randn(5, 32, 192))
    assert logits.shape == (5, 3) and torch.isfinite(logits).all()


def test_unknown_arch():
    with pytest.raises(ValueError):
        build_model({**CONFIG, "arch": "cnn"})
