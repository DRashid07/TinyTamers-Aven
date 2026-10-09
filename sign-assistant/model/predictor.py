"""Load model/artifacts and turn a feature matrix into class probabilities.

Owner: C (Backend/LLM). See CONTRACT.md "Model artifacts".
torch and model.net are imported only when artifacts exist, so the API runs without a model.
"""
import json
import logging
from pathlib import Path

import numpy as np

from pose.features import FEATURE_VERSION, N_FEATURES

log = logging.getLogger("uvicorn.error")


class Predictor:
    def __init__(self, dir):
        """Load config.json and model.pt from dir. Missing or broken artifacts leave it unloaded."""
        self.dir = Path(dir)
        self.config = self.model = None
        config_path, weights_path = self.dir / "config.json", self.dir / "model.pt"
        if not (config_path.exists() and weights_path.exists()):
            log.warning("No model in %s: every recognition abstains with model_not_loaded", self.dir)
            return
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config["feature_version"] != FEATURE_VERSION or config["n_features"] != N_FEATURES:
                raise ValueError(f"features v{config['feature_version']}/{config['n_features']}, "
                                 f"code has v{FEATURE_VERSION}/{N_FEATURES}")
            import torch

            from model.net import build_model
            model = build_model(config)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            model.eval()
            self.config, self.model = config, model
            log.info("Model loaded from %s: %s, %d classes", self.dir, config["arch"], len(config["classes"]))
        except Exception as err:  # noqa: BLE001 - a broken model must not stop the API
            log.warning("Model in %s not loaded (%s): every recognition abstains", self.dir, err)

    @property
    def loaded(self):
        return self.model is not None

    def probs(self, X):
        """X (32, 192) -> np.ndarray (C,) = softmax(logits / temperature)."""
        import torch

        with torch.no_grad():
            logits = self.model(torch.as_tensor(X, dtype=torch.float32)[None])[0]
            return torch.softmax(logits / self.config["temperature"], dim=0).numpy()
