"""Accept a prediction or abstain. Pure function, no torch.

Owner: C (Backend/LLM). See CONTRACT.md "Result".
"""


def decide(probs, info, cfg, vocab):
    """Return a Result dict: status "ok" (id, gloss, confidence) or "abstain" (reason, message)."""
    raise NotImplementedError
