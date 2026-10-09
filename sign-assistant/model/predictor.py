"""Load model/artifacts and turn a feature matrix into class probabilities.

Owner: C (Backend/LLM). See CONTRACT.md "Model artifacts".
"""


class Predictor:
    def __init__(self, dir):
        raise NotImplementedError

    def probs(self, X):
        """X (32, 192) -> np.ndarray (C,) = softmax(logits / temperature)."""
        raise NotImplementedError
