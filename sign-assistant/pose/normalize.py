"""Frame dicts -> landmark arrays, and assignment of hands to the signer's left/right.

Owner: C (Backend/LLM). See CONTRACT.md "Arrays" and "Normalisation".
"""


def frames_to_arrays(frames):
    """Return pose (T,33,4) float32, hands (T,2,21,3) float32 (NaN = missing, unordered), t (T,) ms."""
    raise NotImplementedError


def assign_hands(pose, hands):
    """Give each detected hand to the nearer pose wrist; return left (T,21,3), right (T,21,3)."""
    raise NotImplementedError
