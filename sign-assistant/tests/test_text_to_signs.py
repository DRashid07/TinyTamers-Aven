"""Tests for api/text_to_signs.py with the LLM mocked (no network, no API key).

Owner: D (Direction B/Speech/Eval).
"""
import json

import pytest
from fastapi.testclient import TestClient

from api import text_to_signs as t2s
from api.main import app

VOCAB = [
    {"id": "sabah", "gloss": "SABAH", "az": "sabah", "dataset_label": "SABAH"},
    {"id": "hekim", "gloss": "HƏKİM", "az": "həkim", "dataset_label": "HƏKİM"},
    {"id": "getmek", "gloss": "GETMƏK", "az": "getmək", "dataset_label": "GETMƏK"},
    {"id": "men", "gloss": "MƏN", "az": "mən", "dataset_label": "MƏN"},
    {"id": "menim", "gloss": "MƏNİM", "az": "mənim", "dataset_label": "MƏNİM"},
    {"id": "istemek", "gloss": "İSTƏMƏK", "az": "istəmək", "dataset_label": "İSTƏMƏK"},
    {"id": "bu_gun", "gloss": "BU GÜN", "az": "bu gün", "dataset_label": "BU GÜN"},
    {"id": "ev", "gloss": "EV", "az": "ev", "dataset_label": "EV"},
]


@pytest.fixture
def client(monkeypatch, tmp_path):
    for vocab_id in ("sabah", "hekim", "getmek", "men", "menim", "istemek", "bu_gun"):  # no clip for "ev"
        (tmp_path / f"{vocab_id}.mp4").write_bytes(b"clip")
    monkeypatch.setattr(t2s, "load_vocab", lambda: VOCAB)
    monkeypatch.setattr(t2s, "CLIPS_DIR", tmp_path)
    return TestClient(app)


def llm_answers(monkeypatch, answer):
    """Mock structured_call; returns the list its calls are recorded in."""
    calls = []

    def fake(system, user, schema):
        calls.append((system, user, schema))
        return answer

    monkeypatch.setattr(t2s, "structured_call", fake)
    return calls


def post(client, text):
    response = client.post("/text-to-signs", json={"text": text})
    assert response.status_code == 200
    return response.json()


def test_llm_sequence_unknown_id_becomes_oov_and_clips_are_added(client, monkeypatch):
    calls = llm_answers(monkeypatch, {"sequences": [
        {"sequence": [{"kind": "sign", "id": "xestexana"}]},
        {"sequence": [{"kind": "oov", "word": "Nərgiz"}]}]})
    out = post(client, "Sabah xəstəxanaya evə Nərgiz")
    assert out == {"source": "llm", "sequence": [
        {"kind": "sign", "id": "sabah", "gloss": "SABAH", "clip": "/clips/sabah.mp4"},
        {"kind": "oov", "word": "xestexana"},
        {"kind": "sign", "id": "ev", "gloss": "EV", "clip_missing": True},
        {"kind": "oov", "word": "Nərgiz"}]}
    system, user, schema = calls[0]
    assert "hekim - həkim\ngetmek - getmək" in system and "{vocab_lines}" not in system
    assert '{"kind": "oov", "word": "inkar: getmirəm"}' in system  # the JSON example survived the fill-in
    assert user.endswith('Text spans: ["xəstəxanaya", "Nərgiz"]')
    assert schema["properties"]["sequences"]["items"]["properties"]["sequence"]["items"]["anyOf"][0]["properties"]["id"]["enum"] == [v["id"] for v in VOCAB]


def test_negated_verb_from_the_llm_stays_oov(client, monkeypatch):
    llm_answers(monkeypatch, {"sequence": [
        {"kind": "sign", "id": "getmek"},
        {"kind": "oov", "word": "inkar: getmirəm"}]})
    sequence = post(client, "Sabah həkimə getmirəm.")["sequence"]
    assert [s.get("id") or s["word"] for s in sequence] == ["sabah", "hekim", "getmek", "inkar: getmirəm"]


@pytest.mark.parametrize("answer", [None, {"sequence": []}, {"sequence": [{"kind": "maybe"}]}, {"wrong": 1}])
def test_unusable_llm_answer_uses_the_fallback(client, monkeypatch, answer):
    llm_answers(monkeypatch, answer)
    assert post(client, "naməlum") == {"source": "fallback", "sequence": [{"kind": "oov", "word": "naməlum"}]}


@pytest.mark.parametrize("text, expected", [
    ("Sabah həkimə gedirəm.", ["sabah", "hekim", "getmek"]),
    ("Həkimin", ["hekim"]),
    ("Mənim, mənə!", ["menim", "men"]),
    ("Məndən", ["men"]),
    ("Bu gün evə", ["bu_gun", "ev"]),
    ("ev", ["ev"]),  # a short form still matches exactly
    ("İSTƏMƏK", ["istemek"]),  # İ -> i, not i + combining dot
    ("Sabah həkimə getmirəm", ["sabah", "hekim", "oov:getmirəm"]),  # the negated verb is never getmek
    ("həkim deyil", ["hekim", "oov:deyil"]),
])
def test_fallback_words_and_inflections(client, monkeypatch, text, expected):
    llm_answers(monkeypatch, None)
    out = post(client, text)
    assert out["source"] == "fallback"
    assert [s["id"] if s["kind"] == "sign" else f"oov:{s['word']}" for s in out["sequence"]] == expected


def test_known_words_do_not_depend_on_llm(client, monkeypatch):
    def fail(*args):
        pytest.fail("known words should not request an LLM")

    monkeypatch.setattr(t2s, "structured_call", fail)
    assert post(client, "Bu gün evə gedirəm")["source"] == "fallback"


def test_known_negation_survives_llm_for_other_unmatched_words(client, monkeypatch):
    vocab = VOCAB + [{"id": "yox", "gloss": "YOX", "az": "yox", "dataset_label": "YOX"}]
    monkeypatch.setattr(t2s, "load_vocab", lambda: vocab)
    (t2s.CLIPS_DIR / "yox.mp4").write_bytes(b"clip")
    calls = llm_answers(monkeypatch, {"sequence": [{"kind": "oov", "word": "Nərgiz"}]})
    out = post(client, "Ev deyil, Nərgiz.")
    assert [item.get("id") or item["word"] for item in out["sequence"]] == ["ev", "yox", "Nərgiz"]
    assert calls[0][1] == "Text: Nərgiz"


def test_empty_clip_is_reported_as_missing(client):
    (t2s.CLIPS_DIR / "sabah.mp4").write_bytes(b"")
    assert post(client, "Sabah")["sequence"] == [
        {"kind": "sign", "id": "sabah", "gloss": "SABAH", "clip_missing": True}]


def test_sign_vocab_reports_playable_words(client):
    entries = client.get("/sign-vocab").json()
    assert len(entries) == len(VOCAB)
    assert entries[0]["clip"] == "/clips/sabah.mp4"
    assert entries[-1]["clip_missing"] is True


def test_wrong_batch_size_keeps_known_signs_and_unknown_words(client, monkeypatch):
    calls = llm_answers(monkeypatch, {"sequences": []})
    out = post(client, "Nərgiz sabah xəstəxanaya")
    assert out == {"source": "fallback", "sequence": [
        {"kind": "oov", "word": "Nərgiz"},
        {"kind": "sign", "id": "sabah", "gloss": "SABAH", "clip": "/clips/sabah.mp4"},
        {"kind": "oov", "word": "xəstəxanaya"}]}
    assert len(calls) == 1


def test_greetings_use_extended_playback_vocab(client, monkeypatch):
    vocab = json.loads(t2s.VOCAB_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(t2s, "load_vocab", lambda: vocab)
    for vocab_id in ("salam", "nece", "siz"):
        (t2s.CLIPS_DIR / f"{vocab_id}.mp4").write_bytes(b"clip")
    calls = llm_answers(monkeypatch, None)
    out = post(client, "SALAM, necəsiniz?")
    assert out == {"source": "fallback", "sequence": [
        {"kind": "sign", "id": "salam", "gloss": "SALAM", "clip": "/clips/salam.mp4"},
        {"kind": "sign", "id": "nece", "gloss": "NECƏ", "clip": "/clips/nece.mp4"},
        {"kind": "sign", "id": "siz", "gloss": "SİZ", "clip": "/clips/siz.mp4"}]}
    assert calls == []


def test_playback_vocab_expands_coverage_without_changing_recognition_classes():
    playback = json.loads(t2s.VOCAB_PATH.read_text(encoding="utf-8"))
    recognition = json.loads((t2s.ROOT / "data" / "vocab.json").read_text(encoding="utf-8"))
    assert len(playback) == 100
    assert len({entry["id"] for entry in playback}) == len(playback)
    assert playback[:len(recognition)] == recognition
    assert {"salam", "nece", "yaxsi", "su", "ata", "mekteb"} <= {entry["id"] for entry in playback}


def test_az_lower():
    assert t2s.az_lower("İSTƏMƏK BAKI") == "istəmək bakı"


@pytest.mark.parametrize("text", ["", "   ", "a" * 301])
def test_text_length_is_checked(client, monkeypatch, text):
    llm_answers(monkeypatch, None)
    assert client.post("/text-to-signs", json={"text": text}).status_code == 400
