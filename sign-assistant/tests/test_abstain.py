"""Table-driven tests for model/abstain.py.

Owner: C (Backend/LLM).
Thresholds are exact binary fractions (0.5, 0.25, ...), so "p1 == tau" and "p1 - p2 == margin" are
real boundaries, not float rounding accidents.
"""
import numpy as np
import pytest

from model.abstain import MESSAGE, abstain, decide

CFG = {"classes": ["men", "sen", "hekim"], "tau": 0.5, "margin": 0.25, "min_valid_ratio": 0.5,
       "min_hand_ratio": 0.25}
VOCAB = [{"id": "men", "gloss": "MƏN"}, {"id": "sen", "gloss": "SƏN"}, {"id": "hekim", "gloss": "HƏKİM"}]
GOOD = {"valid_ratio": 1.0, "hand_ratio": 1.0}
REASONS = ["low_confidence", "small_margin", "too_short", "too_long", "no_hands", "invalid_pose",
           "model_not_loaded"]  # every reason in CONTRACT.md "Result"
WORDS = {w for v in VOCAB for w in (v["id"], v["gloss"])}

# (case, probs, info, expected): expected is the class id for "ok", else the abstain reason
CASES = [
    ("clear winner", [0.125, 0.125, 0.75], GOOD, "hekim"),
    ("p1 == tau and p1 - p2 == margin are accepted", [0.5, 0.25, 0.25], GOOD, "men"),
    ("p1 just below tau", [0.4999, 0.25, 0.2501], GOOD, "low_confidence"),
    ("p1 - p2 == margin is accepted", [0.625, 0.375, 0.0], GOOD, "men"),
    ("p1 - p2 just below margin", [0.625, 0.3751, 0.0], GOOD, "small_margin"),
    ("low confidence is checked before the margin", [0.45, 0.44, 0.11], GOOD, "low_confidence"),
    ("valid_ratio == min_valid_ratio is accepted", [0.75, 0.125, 0.125], {"valid_ratio": 0.5, "hand_ratio": 1.0}, "men"),
    ("valid_ratio just below", [0.75, 0.125, 0.125], {"valid_ratio": 0.49, "hand_ratio": 1.0}, "invalid_pose"),
    ("hand_ratio == min_hand_ratio is accepted", [0.75, 0.125, 0.125], {"valid_ratio": 1.0, "hand_ratio": 0.25}, "men"),
    ("hand_ratio just below", [0.75, 0.125, 0.125], {"valid_ratio": 1.0, "hand_ratio": 0.24}, "no_hands"),
    ("invalid pose is checked before no hands", [0.75, 0.125, 0.125], {"valid_ratio": 0.0, "hand_ratio": 0.0},
     "invalid_pose"),
    ("NaN probabilities", [np.nan, 0.5, 0.5], GOOD, "low_confidence"),
    ("infinite probabilities", [np.inf, 0.0, 0.0], GOOD, "low_confidence"),
    ("no probabilities", [], GOOD, "low_confidence"),
]


@pytest.mark.parametrize("case, probs, info, expected", CASES, ids=[c[0] for c in CASES])
def test_decide(case, probs, info, expected):
    result = decide(probs, info, CFG, VOCAB)
    if expected in CFG["classes"]:
        gloss = next(v["gloss"] for v in VOCAB if v["id"] == expected)
        assert result == {"status": "ok", "id": expected, "gloss": gloss, "confidence": max(probs)}
    else:
        assert result == {"status": "abstain", "reason": expected, "message": MESSAGE}


def test_gloss_comes_from_vocab_by_id_not_by_position():
    result = decide([0.0625, 0.875, 0.0625], GOOD, CFG, VOCAB[::-1])
    assert (result["id"], result["gloss"]) == ("sen", "SƏN")


def test_class_missing_from_vocab_abstains():
    assert decide([0.875, 0.0625, 0.0625], GOOD, CFG, VOCAB[1:]) == abstain("model_not_loaded")


@pytest.mark.parametrize("reason", REASONS)
def test_abstain_never_contains_a_word(reason):
    result = abstain(reason)
    assert result == {"status": "abstain", "reason": reason, "message": MESSAGE}
    assert not any(word in str(value) for value in result.values() for word in WORDS)


def test_every_decide_abstain_has_no_word():
    abstains = [decide(p, i, CFG, VOCAB) for _, p, i, e in CASES if e not in CFG["classes"]]
    assert {r["reason"] for r in abstains} == {"low_confidence", "small_margin", "no_hands", "invalid_pose"}
    for result in abstains:
        assert set(result) == {"status", "reason", "message"}
