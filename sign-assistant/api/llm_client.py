"""Claude client shared by /compose-sentence and /text-to-signs.

Owner: C (Backend/LLM). See CONTRACT.md "LLM": official anthropic SDK, model from env
CLAUDE_MODEL (default claude-opus-5-5), effort low, structured outputs, timeout 10 s.
Any error, refusal, timeout or invalid output -> deterministic fallback.
"""
import json
import logging
import os

import anthropic

DEFAULT_MODEL = "claude-opus-5-5"
TIMEOUT_S = 10.0
MAX_RETRIES = 1
MAX_TOKENS = 4096  # small JSON answers; thinking at low effort also counts here

log = logging.getLogger("uvicorn.error")
_client = None


def get_client():
    """The shared client, or None when ANTHROPIC_API_KEY is not set (callers then use their fallback)."""
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


def structured_call(system, user, schema):
    """One Claude request with a JSON-schema answer -> dict, or None (reason logged) on a missing key,
    any API error, timeout, refusal, truncated or schema-invalid output. Never a guess."""
    client = get_client()
    if client is None:
        log.warning("LLM not configured (ANTHROPIC_API_KEY is empty): using the fallback")
        return None
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
    text = next((block.text for block in response.content if block.type == "text"), None)
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        log.warning("LLM returned no valid JSON: using the fallback")
        return None
    if not matches(schema, data):
        log.warning("LLM output does not match the schema: using the fallback")
        return None
    return data
