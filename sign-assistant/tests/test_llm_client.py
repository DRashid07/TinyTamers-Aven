"""Tests for api/llm_client.py with a fake SDK client (no network).

Owner: C (Backend/LLM).
"""
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from api import llm_client

SCHEMA = {"type": "object", "properties": {"sentence": {"type": "string"}, "ok": {"type": "boolean"},
                                           "assumptions": {"type": "array", "items": {"type": "string"}}},
          "required": ["sentence", "ok", "assumptions"], "additionalProperties": False}
GOOD = '{"sentence": "Evə getmək.", "ok": true, "assumptions": ["Kimin getdiyi göstərilməyib."]}'


def reply(text=GOOD, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, stop_details=None,
                           content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)])


class FakeClient:
    def __init__(self, result):
        self.result, self.kwargs = result, None
        self.messages = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def fake(monkeypatch):
    def install(result):
        client = FakeClient(result)
        monkeypatch.setattr(llm_client, "get_client", lambda: client)
        return client
    return install


def test_valid_output_and_request_shape(fake, monkeypatch):
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    client = fake(reply())
    assert llm_client.structured_call("system text", "user text", SCHEMA) == {
        "sentence": "Evə getmək.", "ok": True, "assumptions": ["Kimin getdiyi göstərilməyib."]}
    kw = client.kwargs
    assert kw["model"] == "claude-opus-5-5" and kw["system"] == "system text"
    assert kw["messages"] == [{"role": "user", "content": "user text"}]
    assert kw["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert "thinking" not in kw  # Opus 5.5: thinking is always on, effort is the only control


def test_model_comes_from_env(fake, monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    client = fake(reply())
    llm_client.structured_call("s", "u", SCHEMA)
    assert client.kwargs["model"] == "claude-sonnet-5-5"


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize("result", [
    reply(stop_reason="refusal"),
    reply(stop_reason="max_tokens"),
    reply(text="not json"),
    reply(text='{"sentence": "x", "ok": "yes", "assumptions": []}'),  # wrong type
    reply(text='{"sentence": "x", "ok": true}'),  # missing key
    reply(text='{"sentence": "x", "ok": true, "assumptions": [], "extra": 1}'),  # extra key
    anthropic.APITimeoutError(request=REQUEST),
    anthropic.APIConnectionError(request=REQUEST),
    RuntimeError("anything"),
])
def test_every_failure_returns_none(fake, result):
    fake(result)
    assert llm_client.structured_call("s", "u", SCHEMA) is None


def test_no_key_returns_none_without_a_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm_client, "_client", None)
    assert llm_client.get_client() is None
    assert llm_client.structured_call("s", "u", SCHEMA) is None


def test_client_settings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(llm_client, "_client", None)
    client = llm_client.get_client()
    assert client.max_retries == 1 and client.timeout == 10.0
    assert llm_client.get_client() is client  # one shared client
