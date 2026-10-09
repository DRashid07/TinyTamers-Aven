"""POST /text-to-signs: Azerbaijani text -> sequence of sign clips and out-of-vocabulary words.

Owner: D (Direction B/Speech/Eval). See CONTRACT.md "API" and "LLM".
The LLM maps words to vocab ids; the server checks every item again (an id outside the vocabulary
becomes oov). Without a usable LLM answer a deterministic prefix match is used (source "fallback").
A sign whose clip file is missing gets "clip_missing": true instead of "clip".
"""
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.llm_client import structured_call

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "data" / "vocab.json"
CLIPS_DIR = ROOT / "data" / "clips"
MAX_CHARS = 300
MIN_PREFIX = 3  # an az form shorter than this must match a token exactly

TEXT_TO_SIGNS_SYSTEM_PROMPT = """\
You map Azerbaijani text to a sequence of signs from a FIXED Azerbaijani Sign Language vocabulary. A hearing \
person typed or said the text. A deaf person will watch one video clip per sign, so a wrong or invented sign is \
worse than a missing one.

Vocabulary (id - Azerbaijani meaning):
{vocab_lines}

Rules:
1. Go through the text in order. For each word or fixed phrase that carries meaning, output {"kind": "sign", \
"id": "<id>"} when a vocabulary entry has the same meaning. Inflected forms count: "həkimə", "həkimin" -> həkim; \
"gedirəm", "getdim" -> getmək.
2. Use an entry for a different word only if it is an exact synonym. If you are not sure, output {"kind": "oov", \
"word": "<the word as written>"}.
3. Never output an id that is not in the vocabulary.
4. Negation must never be lost. If a verb is negated (-ma/-mə, "deyil", "yox"), output the verb's sign and then \
the negation sign if the vocabulary has one; if it does not, output {"kind": "oov", "word": "inkar: <original \
word>"}.
5. Skip only grammar words that carry no meaning on their own (for example the particle "ki"). Keep question \
words, numbers, names, times and places; if they are not in the vocabulary, output them as oov.
6. Keep the original word order. Do not apply AzSL grammar; we have no validated rules for it.
7. Return only the JSON object required by the schema.

Example (assuming the vocabulary has sabah, hekim and getmek but no negation sign):
Text: "Sabah həkimə getmirəm."
{"sequence": [{"kind": "sign", "id": "sabah"}, {"kind": "sign", "id": "hekim"}, {"kind": "sign", "id": "getmek"}, \
{"kind": "oov", "word": "inkar: getmirəm"}]}
"""

router = APIRouter()


class TextRequest(BaseModel):
    text: str


def load_vocab():
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


def az_lower(text):
    """Lowercase with Azerbaijani rules: İ -> i and I -> ı (str.lower() gives i + a dot, and i)."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def system_prompt(vocab):
    lines = "\n".join(f"{v['id']} - {v['az']}" for v in vocab)
    return TEXT_TO_SIGNS_SYSTEM_PROMPT.replace("{vocab_lines}", lines)  # .format() would trip on the JSON braces


def schema(ids):
    return {
        "type": "object",
        "properties": {"sequence": {"type": "array", "items": {"anyOf": [
            {"type": "object", "properties": {"kind": {"type": "string", "const": "sign"},
                                              "id": {"type": "string", "enum": ids}},
             "required": ["kind", "id"], "additionalProperties": False},
            {"type": "object", "properties": {"kind": {"type": "string", "const": "oov"},
                                              "word": {"type": "string"}},
             "required": ["kind", "word"], "additionalProperties": False},
        ]}}},
        "required": ["sequence"],
        "additionalProperties": False,
    }


def from_llm(text, vocab):
    """[(kind, id or word)] from the LLM, or None (no LLM, error, malformed or empty answer).
    An id outside the vocabulary becomes oov with the id as the word."""
    ids = [v["id"] for v in vocab]
    answer = structured_call(system_prompt(vocab), f"Text: {text}", schema(ids))
    if answer is None or not isinstance(answer.get("sequence"), list):
        return None
    items = []
    for item in answer["sequence"]:
        kind = item.get("kind") if isinstance(item, dict) else None
        if kind == "sign" and isinstance(item.get("id"), str):
            items.append(("sign", item["id"]) if item["id"] in ids else ("oov", item["id"]))
        elif kind == "oov" and isinstance(item.get("word"), str) and item["word"].strip():
            items.append(("oov", item["word"].strip()))
        else:
            return None
    return items or None


def from_fallback(text, vocab):
    """Deterministic match: each token goes to the vocab entry whose az form is the longest prefix of it
    (at least MIN_PREFIX letters; shorter forms such as "ev" must match exactly). A multi-word form
    ("bu gün") matches a run of tokens. Everything else is oov, written as typed."""
    tokens = re.findall(r"[^\W_]+", text)  # splits on spaces and punctuation
    lowered = [az_lower(t) for t in tokens]
    forms = sorted(((az_lower(v["az"]).split(), v["id"]) for v in vocab), key=lambda f: -len(" ".join(f[0])))
    items, i = [], 0
    while i < len(tokens):
        for words, vocab_id in forms:
            last = i + len(words) - 1
            if last >= len(tokens) or lowered[i:last] != words[:-1]:
                continue
            if lowered[last] == words[-1] or (len(" ".join(words)) >= MIN_PREFIX
                                              and lowered[last].startswith(words[-1])):
                items.append(("sign", vocab_id))
                i = last + 1
                break
        else:
            items.append(("oov", tokens[i]))
            i += 1
    return items


def to_sequence(items, vocab):
    """Contract items: signs get gloss and clip URL (or clip_missing), oov keeps the word."""
    gloss = {v["id"]: v["gloss"] for v in vocab}
    sequence = []
    for kind, value in items:
        if kind == "oov":
            sequence.append({"kind": "oov", "word": value})
        elif (CLIPS_DIR / f"{value}.mp4").exists():
            sequence.append({"kind": "sign", "id": value, "gloss": gloss[value], "clip": f"/clips/{value}.mp4"})
        else:
            sequence.append({"kind": "sign", "id": value, "gloss": gloss[value], "clip_missing": True})
    return sequence


@router.post("/text-to-signs")
def text_to_signs(body: TextRequest):
    """{"text"} (1-300 characters) -> {"sequence": [...], "source": "llm" | "fallback"}; 400 otherwise."""
    text = body.text.strip()
    if not text or len(body.text) > MAX_CHARS:
        raise HTTPException(400, f"send 1 to {MAX_CHARS} characters")
    vocab = load_vocab()
    items = from_llm(text, vocab)
    source = "llm"
    if items is None:
        items, source = from_fallback(text, vocab), "fallback"
    return {"sequence": to_sequence(items, vocab), "source": source}
