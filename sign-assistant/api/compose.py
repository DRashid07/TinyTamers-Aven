"""POST /compose-sentence: recognised sign ids -> Azerbaijani sentence (LLM, else fallback).

Owner: C (Backend/LLM). See CONTRACT.md "API" and "LLM".
"""
from fastapi import APIRouter

router = APIRouter()
