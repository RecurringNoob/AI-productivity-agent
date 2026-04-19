"""
src/llm_client.py — Singleton LLM wrapper for all Gemini calls.

ARCHITECTURE RULE: This is the ONLY file allowed to import from
langchain_google_genai or any LLM SDK. All agents and the supervisor
MUST obtain their LLM instance by calling get_llm(). No other file
may call ChatGoogleGenerativeAI() directly.

Design Document Reference: §3 Architecture ("ALL LLM calls → llm-client")
"""
from __future__ import annotations

import os
from functools import lru_cache

import structlog
from langchain_google_genai import ChatGoogleGenerativeAI

log = structlog.get_logger(__name__)

MODEL_NAME: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
TEMPERATURE: float = 0.0  # Deterministic by design (Design Doc behavioural rules)


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return the singleton ChatGoogleGenerativeAI instance.

    Cached via lru_cache — the model is instantiated exactly once per process.
    Raises RuntimeError if GOOGLE_API_KEY is not set.

    All agents and the supervisor must import only from here:
      from src.llm_client import get_llm
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Copy .env.example → .env and add your Gemini API key."
        )

    log.info("llm_initialised", model=MODEL_NAME, temperature=TEMPERATURE)

    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=TEMPERATURE,
    )
