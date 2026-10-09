"""Tests for model/abstain.py.

Owner: C (Backend/LLM).
"""
import numpy as np
import pytest

from model.abstain import MESSAGE, decide

CFG = {"classes": ["men", "sen", "hekim"], "tau": 0.7, "margin": 0.15, "min_valid_ratio": 0.6,
       "min_hand_ratio": 0.5}
VOCAB = [{"id": "men", "gloss": "MƏN"}, {"id": "sen", "gloss": "SƏN"}, {"id": "hekim", "gloss": "HƏKİM"}]
GOOD = {"valid_ratio": 1.0, "hand_ratio": 1.0}


def test_ok_returns_the_word():
    assert decide([0.05, 0.05, 0.9], GOOD, CFG, VOCAB) == {"status": "ok", "id": "hekim", "gloss": "HƏKİM",
                                                         "confidence": 0.9}


@pytest.mark.parametrize("probs, info, reason", [
    ([0.6, 0.3, 0.1], GOOD, "low_confidence"),  # p1 below tau
    ([0.65, 0.6, 0.0], GOOD, "low_confidence"),  # below tau wins over the small margin
    ([0.0, 0.8, 0.7], GOOD, "small_margin"),  # p1 - p2 = 0.1 < 0.15
    ([0.9, 0.1, 0.0], {"valid_ratio": 0.5, "hand_ratio": 0.0}, "invalid_pose"),  # checked first
    ([0.9, 0.1, 0.0], {"valid_ratio": 0.9, "hand_ratio": 0.4}, "no_hands"),
    ([np.nan, 0.5, 0.5], GOOD, "low_confidence"),
])
def test_abstain_reasons(probs, info, reason):
    result = decide(probs, info, CFG, VOCAB)
    assert result == {"status": "abstain", "reason": reason, "message": MESSAGE}


def test_small_margin_just_below_and_above():
    assert decide([0.80, 0.66, 0.0], GOOD, CFG, VOCAB)["reason"] == "small_margin"  # 0.14 < 0.15
    assert decide([0.80, 0.64, 0.0], GOOD, CFG, VOCAB)["status"] == "ok"  # 0.16


def test_class_missing_from_vocab_abstains():
    assert decide([0.95, 0.05, 0.0], GOOD, CFG, VOCAB[1:])["status"] == "abstain"
