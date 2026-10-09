"""LLM client shared by /compose-sentence and /text-to-signs.

Owner: C (Backend/LLM). See CONTRACT.md "LLM".
- ANTHROPIC_API_KEY set: Claude through the official anthropic SDK, model from CLAUDE_MODEL
  (default claude-opus-5-5), effort low, structured outputs (JSON schema), timeout 10 s, 1 retry.
- Otherwise GROQ_API_KEY set: Groq's free tier, OpenAI-compatible chat completions with a strict JSON schema,
  model from GROQ_MODEL (default openai/gpt-oss-120b), reasoning effort low, timeout 10 s, 1 retry.
  Groq does not store request content by default and its terms forbid training on it; the text still
  leaves the machine.
Any error, refusal, timeout or invalid output -> None, and the caller uses its deterministic fallback.
"""
import json
import logging
import os

import anthropic
import httpx

DEFAULT_MODEL = "claude-opus-5-5"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
TIMEOUT_S = 10.0
MAX_RETRIES = 1
MAX_TOKENS = 4096  # small JSON answers; thinking at low effort also counts here

log = logging.getLogger("uvicorn.error")
_client = None


def get_client():
    """The shared Claude client, or None when ANTHROPIC_API_KEY is not set."""
    global _client
    if _client is None and os.environ.get("ANTHROPIC_API_KEY"):
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=TIMEOUT_S,
                                      max_retries=MAX_RETRIES)
    return _client


def matches(schema, value):
    """True if value fits the small JSON-schema subset we use (object, array, string, boolean)."""
    kind = schema.get("type")
    if kind == "object":
        props = schema.get("properties", {})
        return (isinstance(value, dict) and set(schema.get("required", [])) <= set(value)
                and (schema.get("additionalProperties", True) or set(value) <= set(props))
                and all(matches(props[k], v) for k, v in value.items() if k in props))
    if kind == "array":
        return isinstance(value, list) and all(matches(schema.get("items", {}), v) for v in value)
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    return True


def claude_call(client, system, user, schema):
    """Claude request -> parsed JSON, or None (reason logged)."""
    model = os.environ.get("CLAUDE_MODEL") or DEFAULT_MODEL
    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        )
    except anthropic.APITimeoutError:
        log.warning("LLM timeout after %.0f s (+%d retry): using the fallback", TIMEOUT_S, MAX_RETRIES)
        return None
    except anthropic.APIStatusError as err:
        log.warning("LLM API error %s (%s): using the fallback", err.status_code, type(err).__name__)
        return None
    except anthropic.APIConnectionError as err:
        log.warning("LLM connection error (%s): using the fallback", err)
        return None
    except Exception as err:  # noqa: BLE001 - nothing may break the endpoint
        log.warning("LLM call failed (%s): using the fallback", type(err).__name__)
        return None
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        log.warning("LLM refused (category %s): using the fallback", getattr(details, "category", None))
        return None
    if response.stop_reason != "end_turn":
        log.warning("LLM stopped with %s: using the fallback", response.stop_reason)
        return None
    return parse_json(next((block.text for block in response.content if block.type == "text"), None))


def groq_call(key, system, user, schema):
    """Groq (OpenAI-compatible) request with a strict JSON schema -> parsed JSON, or None (reason logged)."""
    model = os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "answer", "strict": True, "schema": schema}}}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = httpx.post(GROQ_URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT_S)
        except httpx.TimeoutException:
            log.warning("Groq timeout after %.0f s (attempt %d)", TIMEOUT_S, attempt + 1)
            continue
        except httpx.HTTPError as err:
            log.warning("Groq connection error (%s): using the fallback", type(err).__name__)
            return None
        if response.status_code in (408, 429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
            continue
        if response.status_code != 200:
            log.warning("Groq API error %s: using the fallback", response.status_code)
            return None
        try:
            choice = response.json()["choices"][0]
        except (ValueError, KeyError, IndexError):
            log.warning("Groq returned an unexpected body: using the fallback")
            return None
        if choice.get("message", {}).get("refusal"):
            log.warning("Groq model refused: using the fallback")
            return None
        if choice.get("finish_reason") != "stop":
            log.warning("Groq stopped with %s: using the fallback", choice.get("finish_reason"))
            return None
        return parse_json(choice.get("message", {}).get("content"))
    log.warning("Groq failed after %d attempts: using the fallback", MAX_RETRIES + 1)
    return None


def parse_json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        log.warning("LLM returned no valid JSON: using the fallback")
        return None


def structured_call(system, user, schema):
    """One LLM request with a JSON-schema answer -> dict, or None (reason logged) on a missing key, any API
    error, timeout, refusal, truncated or schema-invalid output. Never a guess. Claude first, then Groq."""
    client = get_client()
    if client is not None:
        data = claude_call(client, system, user, schema)
    elif os.environ.get("GROQ_API_KEY"):
        data = groq_call(os.environ["GROQ_API_KEY"], system, user, schema)
    else:
        log.warning("LLM not configured (ANTHROPIC_API_KEY and GROQ_API_KEY are empty): using the fallback")
        return None
    if data is None:
        return None
    if not matches(schema, data):
        log.warning("LLM output does not match the schema: using the fallback")
        return None
    return data
