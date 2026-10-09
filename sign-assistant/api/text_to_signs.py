"""POST /text-to-signs: Azerbaijani text -> sequence of sign clips and out-of-vocabulary words.

Owner: D (Direction B/Speech/Eval). See CONTRACT.md "API" and "LLM".
"""
from fastapi import APIRouter

router = APIRouter()
