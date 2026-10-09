"""POST /compose-sentence: recognised sign ids -> Azerbaijani sentence (LLM, else fallback).

Owner: C (Backend/LLM). See CONTRACT.md "API" and "LLM".
Review table for a native speaker (docs/LLM_CHECK.md): python -m api.compose --check
"""
import argparse
import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.llm_client import structured_call

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "data" / "vocab.json"
MAX_IDS = 12

# The task's prompt, with its three examples rewritten in words from data/vocab.json as it asked.
COMPOSE_SYSTEM_PROMPT = """\
You turn a sequence of Azerbaijani Sign Language (AzSL) glosses into one short, grammatical Azerbaijani sentence. \
The glosses come from an assistive prototype used at clinics and public offices, where an added detail can cause \
real harm. Accuracy matters more than fluency.

Rules:
1. Every piece of meaning in the sentence must come from the glosses. Do not add any person, object, place, time, \
date, number, symptom, body part, cause, negation, question, request, wish, politeness formula or emotion that the \
glosses do not express.
2. You may only reorder the glosses, add inflectional suffixes (case, possessive, person, number, tense, mood), and \
add the smallest function word or copula that Azerbaijani grammar requires.
3. Use every gloss. Do not drop or merge any.
4. If no gloss gives the time, use the present tense. If no gloss gives the subject, do not choose a person: prefer \
an infinitive phrase (for example "Evə getmək."). Record every grammatical choice the glosses did not determine in \
"assumptions", written in Azerbaijani.
5. If the glosses cannot form a sentence without breaking rule 1 (for example, the relation between them is \
unclear), set "ok" to false and return the glosses' Azerbaijani meanings in the original order, lowercase, \
separated by spaces.
6. Write in the Azerbaijani Latin alphabet. Return only the JSON object required by the schema.

Examples:
Glosses: MƏN (mən) | BAKI (bakı) | GETMƏK (getmək) | İSTƏMƏK (istəmək)
{"sentence": "Mən Bakıya getmək istəyirəm.", "ok": true, "assumptions": []}

Glosses: EV (ev) | GETMƏK (getmək)
{"sentence": "Evə getmək.", "ok": true, "assumptions": ["Kimin getdiyi göstərilməyib."]}

Glosses: SABAH (sabah) | TELEFON (telefon) | PENSİYA (pensiya)
{"sentence": "sabah telefon pensiya", "ok": false, "assumptions": ["Sözlər arasındakı əlaqə aydın deyil."]}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "sentence": {"type": "string"},
        "ok": {"type": "boolean"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sentence", "ok", "assumptions"],
    "additionalProperties": False,
}

router = APIRouter()


class ComposeRequest(BaseModel):
    ids: list[str]


def load_vocab():
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


def user_message(entries):
    return "Glosses in signed order: " + " | ".join(f"{e['gloss']} ({e['az']})" for e in entries)


def compose(entries):
    """Vocab entries in signed order -> contract response. The LLM's answer is used only if it is
    schema-valid; otherwise the az meanings joined by spaces, ok false, source "fallback"."""
    glosses = [e["gloss"] for e in entries]
    answer = structured_call(COMPOSE_SYSTEM_PROMPT, user_message(entries), SCHEMA)
    if answer is None or not answer["sentence"].strip():
        return {"sentence": " ".join(e["az"] for e in entries), "ok": False, "assumptions": [],
                "glosses": glosses, "source": "fallback"}
    return {"sentence": answer["sentence"].strip(), "ok": answer["ok"], "assumptions": answer["assumptions"],
            "glosses": glosses, "source": "llm"}


@router.post("/compose-sentence")
def compose_sentence(body: ComposeRequest):
    """{"ids": [...]} -> {"sentence", "ok", "assumptions", "glosses", "source"}; 400 on unknown ids,
    no ids or more than MAX_IDS."""
    if not body.ids or len(body.ids) > MAX_IDS:
        raise HTTPException(400, f"send 1 to {MAX_IDS} ids")
    by_id = {v["id"]: v for v in load_vocab()}
    unknown = [i for i in body.ids if i not in by_id]
    if unknown:
        raise HTTPException(400, f"unknown ids: {unknown}")
    return compose([by_id[i] for i in body.ids])


# 10 review lists from data/vocab.json; the last 3 cannot form a sentence without adding information.
CHECK_LISTS = [
    ["men", "sabah", "baki", "getmek", "istemek"],
    ["men", "sened", "almaq", "istemek"],
    ["menim", "usaq", "var"],
    ["men", "elil", "pensiya", "almaq", "istemek"],
    ["o", "dunen", "ev", "getmek"],
    ["bu_gun", "is", "yox"],
    ["siz", "harda"],
    ["telefon", "futbol", "viza"],
    ["sabah", "ana", "etibarname"],
    ["d", "saglam", "orda"],
]


def main(argv=None):
    p = argparse.ArgumentParser(description="Sentence composition review table.")
    p.add_argument("--check", action="store_true", help="compose the 10 review lists and print the table")
    a = p.parse_args(argv)
    if not a.check:
        p.print_help()
        return
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    by_id = {v["id"]: v for v in load_vocab()}
    print("# | glosses | sentence | ok | source | assumptions")
    for n, ids in enumerate(CHECK_LISTS, 1):
        out = compose([by_id[i] for i in ids])
        print(f"{n} | {' '.join(out['glosses'])} | {out['sentence']} | {out['ok']} | {out['source']} | "
              f"{'; '.join(out['assumptions']) or '-'}")
    print("Lists 8-10 should come back ok = False. A native speaker reviews this table in docs/LLM_CHECK.md.")


if __name__ == "__main__":
    main()
