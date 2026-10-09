"""Claude client shared by /compose-sentence and /text-to-signs.

Owner: C (Backend/LLM). See CONTRACT.md "LLM": official anthropic SDK, model from env
CLAUDE_MODEL (default claude-opus-5-5), effort low, structured outputs, timeout 10 s.
Any error, refusal, timeout or invalid output -> deterministic fallback.
"""
