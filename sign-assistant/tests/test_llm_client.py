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
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_client, "_client", None)
    assert llm_client.get_client() is None
    assert llm_client.structured_call("s", "u", SCHEMA) is None


def test_client_settings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(llm_client, "_client", None)
    client = llm_client.get_client()
    assert client.max_retries == 1 and client.timeout == 10.0
    assert llm_client.get_client() is client  # one shared client


# --- Groq (free tier) path: used only when there is no Anthropic key ---

class FakeHttp:
    """Stands in for httpx.post: returns the queued (status, body) answers or raises queued exceptions."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        return fake_response(status, body)


def fake_response(status, body):
    import httpx
    return httpx.Response(status, json=body, request=httpx.Request("POST", llm_client.GROQ_URL))


def groq_ok(content=GOOD, finish="stop", refusal=None):
    return 200, {"choices": [{"finish_reason": finish, "message": {"content": content, "refusal": refusal}}]}


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setattr(llm_client, "get_client", lambda: None)  # no Anthropic key
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-not-real")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    def install(*answers):
        fake_post = FakeHttp(*answers)
        monkeypatch.setattr(llm_client.httpx, "post", fake_post)
        return fake_post
    return install


def test_groq_valid_output_and_request_shape(groq):
    post = groq(groq_ok())
    assert llm_client.structured_call("system text", "user text", SCHEMA)["sentence"] == "Evə getmək."
    call = post.calls[0]
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions" and call["timeout"] == 10.0
    assert call["headers"] == {"Authorization": "Bearer gsk-test-not-real"}
    body = call["json"]
    assert body["model"] == "openai/gpt-oss-120b" and body["reasoning_effort"] == "low"
    assert body["messages"] == [{"role": "system", "content": "system text"}, {"role": "user", "content": "user text"}]
    assert body["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "answer", "strict": True, "schema": SCHEMA}}


def test_groq_retries_once_on_429_and_timeout(groq):
    import httpx
    post = groq((429, {"error": "rate"}), groq_ok())
    assert llm_client.structured_call("s", "u", SCHEMA) is not None and len(post.calls) == 2
    post = groq(httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"))
    assert llm_client.structured_call("s", "u", SCHEMA) is None and len(post.calls) == 2


@pytest.mark.parametrize("answer", [
    groq_ok(finish="length"),
    groq_ok(refusal="I can't help with that."),
    groq_ok(content="not json"),
    groq_ok(content='{"sentence": "x", "ok": true}'),  # schema-invalid
    (400, {"error": {"message": "Generated JSON does not match the expected schema."}}),
    (401, {"error": "bad key"}),
    (200, {"unexpected": True}),
])
def test_groq_failures_return_none(groq, answer):
    groq(answer)
    assert llm_client.structured_call("s", "u", SCHEMA) is None


def test_claude_is_used_when_both_keys_exist(fake, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-not-real")
    client = fake(reply())
    monkeypatch.setattr(llm_client.httpx, "post", lambda *a, **k: pytest.fail("Groq must not be called"))
    assert llm_client.structured_call("s", "u", SCHEMA) is not None and client.kwargs is not None
