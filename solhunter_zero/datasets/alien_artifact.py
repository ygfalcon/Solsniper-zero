"""Alien artifact pattern dataset loader."""
from __future__ import annotations

import json
import os
from typing import List, Dict, Any

# Path to the dataset JSON file relative to this module
_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "alien_artifact_patterns.json")

_patterns: List[Dict[str, Any]] | None = None


def _load_dataset() -> List[Dict[str, Any]]:
    with open(_DATA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_patterns() -> List[Dict[str, Any]]:
    """Return list of alien artifact patterns."""
    global _patterns
    if _patterns is None:
        _patterns = _load_dataset()
    return _patterns


def get_encoding_by_glyphs(glyphs: str) -> List[int] | None:
    """Return the encoding list for the given glyph sequence or ``None``."""
    for entry in load_patterns():
        if entry.get("glyphs") == glyphs:
            enc = entry.get("encoding")
            if isinstance(enc, list):
                return enc
    return None

