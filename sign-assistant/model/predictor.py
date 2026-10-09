"""Load model/artifacts and turn a feature matrix into class probabilities.

Owner: C (Backend/LLM). See CONTRACT.md "Model artifacts".
torch and model.net are imported only when artifacts exist, so the API runs without a model.
"""
import json
import logging
import time
from pathlib import Path

import numpy as np

from pose.features import FEATURE_VERSION, N_FEATURES

log = logging.getLogger("uvicorn.error")
REQUIRED = ("T", "arch", "classes", "temperature", "tau", "margin", "min_valid_ratio", "min_hand_ratio")


def check_config(config):
    """Raise ValueError unless config matches the feature code and its abstain settings make sense."""
    if config.get("feature_version") != FEATURE_VERSION or config.get("n_features") != N_FEATURES:
        raise ValueError(f"features v{config.get('feature_version')}/{config.get('n_features')}, "
                         f"code has v{FEATURE_VERSION}/{N_FEATURES}")
    missing = [key for key in REQUIRED if key not in config]
    if missing:
        raise ValueError(f"config.json lacks {missing}")
    if (not config["classes"] or config["temperature"] <= 0 or not 0 < config["tau"] <= 1
            or config["margin"] < 0):
        raise ValueError("config.json needs classes, temperature > 0, 0 < tau <= 1 and margin >= 0")


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
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))  # tolerate a BOM from Windows editors
            check_config(config)
            import torch

            from model.net import build_model
            model = build_model(config)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            model.eval()
            start = time.perf_counter()
            with torch.no_grad():  # warm-up: the first forward pass is the slowest
                model(torch.zeros(1, config["T"], N_FEATURES))
            warm_up_ms = (time.perf_counter() - start) * 1000
            self.config, self.model = config, model
            log.info("Model loaded from %s: %s, %d classes, temperature %s, tau %s, margin %s "
                     "(warm-up inference %.1f ms)", self.dir, config["arch"], len(config["classes"]),
                     config["temperature"], config["tau"], config["margin"], warm_up_ms)
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
