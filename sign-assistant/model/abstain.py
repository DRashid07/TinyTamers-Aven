"""Accept a prediction or abstain. Pure function, no torch.

Owner: C (Backend/LLM). See CONTRACT.md "Result".
"""
import numpy as np

MESSAGE = "Əmin deyiləm, zəhmət olmasa təkrar edin."


def abstain(reason):
    """Abstain Result. It never carries a word (no id, gloss or confidence)."""
    return {"status": "abstain", "reason": reason, "message": MESSAGE}


def decide(probs, info, cfg, vocab):
    """Return a Result dict: status "ok" (id, gloss, confidence) or "abstain" (reason, message).

    Checks in order: invalid_pose, no_hands, low_confidence (p1 < tau), small_margin (p1 - p2 < margin).
    cfg is model/artifacts/config.json; class i of probs is cfg["classes"][i].
    """
    if info["valid_ratio"] < cfg["min_valid_ratio"]:
        return abstain("invalid_pose")
    if info["hand_ratio"] < cfg["min_hand_ratio"]:
        return abstain("no_hands")
    probs = np.asarray(probs, dtype=np.float64)
    if probs.size == 0 or not np.isfinite(probs).all():
        return abstain("low_confidence")
    order = np.argsort(probs)[::-1]
    p1 = probs[order[0]]
    p2 = probs[order[1]] if probs.size > 1 else 0.0
    if p1 < cfg["tau"]:
        return abstain("low_confidence")
    if p1 - p2 < cfg["margin"]:
        return abstain("small_margin")
    class_id = cfg["classes"][order[0]]
    gloss = next((v["gloss"] for v in vocab if v["id"] == class_id), None)
    if gloss is None:  # model and vocab.json disagree: never show a word we cannot name
        return abstain("model_not_loaded")
    return {"status": "ok", "id": class_id, "gloss": gloss, "confidence": round(float(p1), 4)}
