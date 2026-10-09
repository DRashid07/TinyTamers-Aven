"""POST /text-to-signs: Azerbaijani text -> sequence of sign clips and out-of-vocabulary words.

Owner: D (Direction B/Speech/Eval). See CONTRACT.md "API" and "LLM".
Known words and inflections use deterministic matches from the playback vocabulary. The LLM only
resolves unmatched spans; the server checks every returned id against the same vocabulary.
A sign whose clip file is missing gets "clip_missing": true instead of "clip".
"""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.llm_client import structured_call
from api.sign_matching import az_lower, from_fallback

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "data" / "playback_vocab.json"
CLIPS_DIR = ROOT / "data" / "clips"
MAX_CHARS = 300

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


def validate_answer(answer, vocab):
    """Validate an LLM sequence; an id outside the vocabulary becomes an OOV word."""
    ids = [v["id"] for v in vocab]
    if not isinstance(answer, dict) or not isinstance(answer.get("sequence"), list):
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


def from_llm(text, vocab):
    """Resolve an unmatched text span, or return None on an unusable LLM answer."""
    answer = structured_call(system_prompt(vocab), f"Text: {text}", schema([v["id"] for v in vocab]))
    return validate_answer(answer, vocab)


def resolve_unknown(items, vocab):
    """Keep known matches in order; resolve all unmatched spans with at most one LLM request."""
    spans, i = [], 0
    while i < len(items):
        if items[i][0] != "oov":
            i += 1
            continue
        start = i
        while i < len(items) and items[i][0] == "oov":
            i += 1
        spans.append((start, i, " ".join(word for _, word in items[start:i])))
    if not spans:
        return items, "fallback"
    if len(spans) == 1:
        replacements = [from_llm(spans[0][2], vocab)]
    else:
        batch_schema = {"type": "object", "properties": {"sequences": {
            "type": "array", "items": schema([v["id"] for v in vocab])}},
            "required": ["sequences"], "additionalProperties": False}
        user = ("Resolve these unmatched text spans independently, in the given order. Return one "
                "sequence per span in the sequences array. Do not combine spans.\nText spans: "
                + json.dumps([span[2] for span in spans], ensure_ascii=False))
        answer = structured_call(system_prompt(vocab), user, batch_schema)
        sequences = answer.get("sequences") if isinstance(answer, dict) else None
        if not isinstance(sequences, list) or len(sequences) != len(spans):
            return items, "fallback"
        replacements = [validate_answer(sequence, vocab) for sequence in sequences]
    out, start, used_llm = [], 0, False
    for (left, right, _), replacement in zip(spans, replacements):
        out.extend(items[start:left])
        out.extend(replacement if replacement is not None else items[left:right])
        used_llm |= replacement is not None
        start = right
    out.extend(items[start:])
    return out, "llm" if used_llm else "fallback"


def clip_fields(vocab_id):
    path = CLIPS_DIR / f"{vocab_id}.mp4"
    if path.is_file() and path.stat().st_size:
        return {"clip": f"/clips/{vocab_id}.mp4"}
    return {"clip_missing": True}


@router.get("/sign-vocab")
def sign_vocab():
    """The full playback vocabulary and available clips; /vocab remains the recognition classes."""
    return [{**entry, **clip_fields(entry["id"])} for entry in load_vocab()]


def to_sequence(items, vocab):
    """Contract items: signs get gloss and clip URL (or clip_missing), oov keeps the word."""
    gloss = {v["id"]: v["gloss"] for v in vocab}
    sequence = []
    for kind, value in items:
        if kind == "oov":
            sequence.append({"kind": "oov", "word": value})
        else:
            sequence.append({"kind": "sign", "id": value, "gloss": gloss[value], **clip_fields(value)})
    return sequence


@router.post("/text-to-signs")
def text_to_signs(body: TextRequest):
    """{"text"} (1-300 characters) -> {"sequence": [...], "source": "llm" | "fallback"}; 400 otherwise."""
    text = body.text.strip()
    if not text or len(body.text) > MAX_CHARS:
        raise HTTPException(400, f"send 1 to {MAX_CHARS} characters")
    vocab = load_vocab()
    items, source = resolve_unknown(from_fallback(text, vocab), vocab)
    return {"sequence": to_sequence(items, vocab), "source": source}
