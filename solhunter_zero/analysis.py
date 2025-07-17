"""Utility helpers for light-weight AI analysis."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")


def ask_llm(prompt: str, *, model: str | None = None) -> str:
    """Send ``prompt`` to a local ollama server and return the response text."""

    payload: Dict[str, Any] = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", "")).strip()
    except Exception as exc:  # network issues, JSON decoding etc.
        logger.error("LLM request failed: %s", exc)
        return ""


def sentiment_score(token_name: str) -> float:
    """Get a rough sentiment score from the LLM about ``token_name``."""

    prompt = (
        "On a scale from -1 to 1, how exciting does this Solana token name sound: "
        f"'{token_name}'? Reply with just the number."
    )
    reply = ask_llm(prompt)
    try:
        # extract the first float from the reply
        return float(reply.split()[0])
    except Exception:
        logger.error("Failed to parse sentiment from: %r", reply)
        return 0.0
