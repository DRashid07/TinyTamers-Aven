"""Tests for api/speech.py with httpx mocked (no network, no Azure key).

Owner: D (Direction B/Speech/Eval).
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from api import speech
from api.main import app

MP3 = b"ID3\x04\x00fake-mp3-bytes"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "westeurope")
    monkeypatch.delenv("AZURE_SPEECH_VOICE", raising=False)
    speech._cache.clear()
    return TestClient(app)


def azure(monkeypatch, reply):
    """Mock httpx.post; reply is an httpx.Response or an exception. Returns the recorded calls."""
    calls = []

    def fake_post(url, content, headers, timeout):
        calls.append({"url": url, "content": content.decode("utf-8"), "headers": headers, "timeout": timeout})
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(speech.httpx, "post", fake_post)
    return calls


def test_missing_key_is_503_and_azure_is_not_called(client, monkeypatch):
    monkeypatch.delenv("AZURE_SPEECH_KEY")
    calls = azure(monkeypatch, httpx.Response(200, content=MP3))
    response = client.post("/tts", json={"text": "Salam"})
    assert response.status_code == 503 and "AZURE_SPEECH_KEY" in response.json()["detail"]
    assert calls == []


def test_bad_region_is_503(client, monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_REGION", "west europe/x")
    azure(monkeypatch, httpx.Response(200, content=MP3))
    assert client.post("/tts", json={"text": "Salam"}).status_code == 503


def test_success_returns_mp3_and_sends_ssml(client, monkeypatch):
    calls = azure(monkeypatch, httpx.Response(200, content=MP3))
    response = client.post("/tts", json={"text": "Salam & <xoş> gəldiniz"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg" and response.content == MP3
    (call,) = calls
    assert call["url"] == "https://westeurope.tts.speech.microsoft.com/cognitiveservices/v1"
    assert call["headers"]["Ocp-Apim-Subscription-Key"] == "test-key"
    assert call["headers"]["X-Microsoft-OutputFormat"] == "audio-24khz-48kbitrate-mono-mp3"
    assert call["headers"]["Content-Type"] == "application/ssml+xml" and call["timeout"] == 8.0
    assert "<voice name='az-AZ-BanuNeural'>Salam &amp; &lt;xoş&gt; gəldiniz</voice>" in call["content"]


def test_same_text_is_served_from_the_cache(client, monkeypatch):
    calls = azure(monkeypatch, httpx.Response(200, content=MP3))
    for _ in range(3):
        assert client.post("/tts", json={"text": "Salam"}).content == MP3
    assert len(calls) == 1


def test_cache_drops_the_least_recently_used(client, monkeypatch):
    monkeypatch.setattr(speech, "CACHE_SIZE", 2)
    calls = azure(monkeypatch, httpx.Response(200, content=MP3))
    for text in ["a", "b", "a", "c", "a", "b"]:  # "a" is used again before "c", so "b" goes first
        client.post("/tts", json={"text": text})
    assert [c["content"].split(">")[2][0] for c in calls] == ["a", "b", "c", "b"]


@pytest.mark.parametrize("text", ["x" * 301, "", "   "])
def test_bad_text_is_422(client, monkeypatch, text):
    calls = azure(monkeypatch, httpx.Response(200, content=MP3))
    assert client.post("/tts", json={"text": text}).status_code == 422
    assert calls == []


@pytest.mark.parametrize("reply, status", [
    (httpx.Response(401, content=b"unauthorized"), 502),
    (httpx.Response(200, content=b""), 502),
    (httpx.ConnectError("no route"), 502),
    (httpx.ReadTimeout("slow"), 504),
])
def test_azure_failures(client, monkeypatch, reply, status):
    azure(monkeypatch, reply)
    response = client.post("/tts", json={"text": "Salam"})
    assert response.status_code == status and "Azure" in response.json()["detail"]
