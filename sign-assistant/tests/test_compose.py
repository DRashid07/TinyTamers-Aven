"""Tests for api/compose.py with the LLM mocked (no network, no key needed).

Owner: C (Backend/LLM).
"""
import pytest
from fastapi.testclient import TestClient

from api import compose
from api.main import app

VOCAB = [{"id": "men", "gloss": "MƏN", "az": "mən", "dataset_label": "MƏN"},
         {"id": "baki", "gloss": "BAKI", "az": "bakı", "dataset_label": "BAKI"},
         {"id": "getmek", "gloss": "GETMƏK", "az": "getmək", "dataset_label": "GETMƏK"},
         {"id": "istemek", "gloss": "İSTƏMƏK", "az": "istəmək", "dataset_label": "İSTƏMƏK"}]
IDS = ["men", "baki", "getmek", "istemek"]


@pytest.fixture
def client(monkeypatch):
    calls = []
    monkeypatch.setattr(compose, "load_vocab", lambda: VOCAB)
    monkeypatch.setattr(compose, "structured_call", lambda s, u, schema: calls.append((s, u, schema)) or None)
    c = TestClient(app)
    c.calls = calls
    return c


def post(client, ids):
    return client.post("/compose-sentence", json={"ids": ids})


def test_unknown_id_is_400(client):
    response = post(client, ["men", "hekim"])
    assert response.status_code == 400 and "hekim" in response.json()["detail"]
    assert client.calls == []  # the LLM is never asked


@pytest.mark.parametrize("ids", [[], ["men"] * 13])
def test_empty_or_too_many_ids_is_400(client, ids):
    assert post(client, ids).status_code == 400


def test_llm_none_gives_the_fallback(client):
    assert post(client, IDS).json() == {"sentence": "mən bakı getmək istəmək", "ok": False, "assumptions": [],
                                        "glosses": ["MƏN", "BAKI", "GETMƏK", "İSTƏMƏK"], "source": "fallback"}


def test_valid_llm_output_passes_through(client, monkeypatch):
    answer = {"sentence": "Mən Bakıya getmək istəyirəm.", "ok": True, "assumptions": []}
    monkeypatch.setattr(compose, "structured_call", lambda s, u, schema: answer)
    assert post(client, IDS).json() == {**answer, "glosses": ["MƏN", "BAKI", "GETMƏK", "İSTƏMƏK"],
                                        "source": "llm"}


def test_empty_llm_sentence_falls_back(client, monkeypatch):
    monkeypatch.setattr(compose, "structured_call", lambda s, u, schema: {"sentence": " ", "ok": True,
                                                                          "assumptions": []})
    assert post(client, IDS).json()["source"] == "fallback"


def test_request_to_the_llm(client):
    post(client, ["getmek", "men", "baki"])
    system, user, schema = client.calls[0]
    assert system == compose.COMPOSE_SYSTEM_PROMPT
    assert user == "Glosses in signed order: GETMƏK (getmək) | MƏN (mən) | BAKI (bakı)"  # signed order kept
    assert all(f"{v['gloss']} ({v['az']})" in user for v in VOCAB[:3])
    assert schema["additionalProperties"] is False and schema["required"] == ["sentence", "ok", "assumptions"]


def test_prompt_and_check_lists_use_only_our_vocabulary():
    prompt = compose.COMPOSE_SYSTEM_PROMPT
    assert "Accuracy matters more than fluency." in prompt and "Return only the JSON object" in prompt
    assert "\\" not in prompt and "\n\n" in prompt  # line joins left no backslashes behind
    vocab = {v["id"] for v in compose.load_vocab()}
    assert len(compose.CHECK_LISTS) == 10 and all(set(ids) <= vocab for ids in compose.CHECK_LISTS)
    assert all(len(ids) <= compose.MAX_IDS for ids in compose.CHECK_LISTS)
