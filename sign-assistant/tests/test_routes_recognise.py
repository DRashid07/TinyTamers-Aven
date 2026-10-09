"""Tests for api/routes_recognise.py with synthetic landmark frames.

Owner: C (Backend/LLM).
"""
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import routes_recognise
from api.main import app
from model.predictor import Predictor
from tests.fake_pose import H, W, raise_both, sequence

SIGN = sequence((10, {}), (15, raise_both()), (10, {}))  # down -> up -> down, 467 ms raised
VOCAB = [{"id": "men", "gloss": "MƏN", "az": "mən", "dataset_label": "MƏN"},
         {"id": "sen", "gloss": "SƏN", "az": "sən", "dataset_label": "SƏN"}]


class FakePredictor:
    loaded = True
    config = {"T": 32, "arch": "gru", "classes": ["men", "sen"], "temperature": 1.0, "tau": 0.7,
              "margin": 0.15, "min_valid_ratio": 0.6, "min_hand_ratio": 0.5}

    def __init__(self, probs):
        self._probs = np.array(probs)

    def probs(self, X):
        assert X.shape == (32, 192)
        return self._probs


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(routes_recognise, "PREDICTOR", Predictor(tmp_path))  # no artifacts -> not loaded
    monkeypatch.setattr(routes_recognise, "load_vocab", lambda: VOCAB)
    return TestClient(app)


def ws_messages(client, frames):
    with client.websocket_connect("/ws/recognise") as ws:
        ws.send_json({"type": "start", "w": W, "h": H})
        for f in frames:
            ws.send_json(f)
        out = []
        while not out or out[-1]["type"] != "result":
            out.append(ws.receive_json())
    return out


def test_ws_up_down_gives_state_then_result(client):
    msgs = ws_messages(client, SIGN)
    assert msgs[:2] == [{"type": "state", "signing": True}, {"type": "state", "signing": False}]
    assert msgs[2]["type"] == "result" and msgs[2]["status"] == "abstain"
    assert msgs[2]["reason"] == "model_not_loaded" and msgs[2]["message"].startswith("Əmin deyiləm")
    assert "gloss" not in msgs[2] and "id" not in msgs[2]  # never a word when abstaining


def test_ws_too_short_and_bad_messages(client):
    with client.websocket_connect("/ws/recognise") as ws:
        ws.send_text("not json")
        ws.send_json({"type": "frame", "t": 0, "pose": [[1, 2]], "hands": []})  # before the header, ignored
        ws.send_json({"type": "start", "w": W, "h": H})
        ws.send_json({"type": "frame", "t": 1, "pose": [[1, 2]], "hands": []})  # wrong shape, dropped
        for f in sequence((5, {}), (5, raise_both()), (10, {})):
            ws.send_json(f)
        msgs = [ws.receive_json() for _ in range(3)]
    assert msgs[2] == {"type": "result", "status": "abstain", "reason": "too_short",
                       "message": msgs[2]["message"]}


def test_ws_ok_result_with_a_model(client, monkeypatch):
    monkeypatch.setattr(routes_recognise, "PREDICTOR", FakePredictor([0.9, 0.1]))
    result = ws_messages(client, SIGN)[-1]
    assert result == {"type": "result", "status": "ok", "id": "men", "gloss": "MƏN", "confidence": 0.9}


def test_post_without_model_returns_model_not_loaded(client):
    record = {"id": "men", "signer": "test", "w": W, "h": H, "frames": SIGN}  # a record-mode file
    response = client.post("/recognise", json=record)
    assert response.status_code == 200
    assert response.json()["reason"] == "model_not_loaded"


def test_post_without_raised_hands_and_bad_input(client):
    still = {"w": W, "h": H, "frames": sequence((20, {}))}
    assert client.post("/recognise", json=still).json()["reason"] == "no_hands"
    assert client.post("/recognise", json={"h": H, "frames": []}).status_code == 422
    bad = {"w": W, "h": H, "frames": [{"type": "frame", "t": 0, "pose": [[1]], "hands": []}]}
    assert client.post("/recognise", json=bad).status_code == 422


def test_post_ok_and_unsegmented(client, monkeypatch):
    monkeypatch.setattr(routes_recognise, "PREDICTOR", FakePredictor([0.2, 0.8]))
    assert client.post("/recognise", json={"w": W, "h": H, "frames": SIGN}).json()["gloss"] == "SƏN"
    whole = client.post("/recognise", json={"w": W, "h": H, "frames": SIGN, "segment": False}).json()
    assert whole["id"] == "sen"


def test_latency_over_budget_is_a_warning(client, monkeypatch, caplog):
    monkeypatch.setattr(routes_recognise, "PREDICTOR", FakePredictor([0.9, 0.1]))
    monkeypatch.setattr(routes_recognise, "LATENCY_BUDGET_MS", -1)
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        client.post("/recognise", json={"w": W, "h": H, "frames": SIGN})
    assert any(r.levelno == logging.WARNING and "latency" in r.getMessage() for r in caplog.records)


def test_vocab_and_model_info(client):
    assert client.get("/vocab").json() == VOCAB
    assert client.get("/model-info").json() == {"loaded": False, "arch": None, "n_classes": 0, "tau": None,
                                                "margin": None, "temperature": None}
