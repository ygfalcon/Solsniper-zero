"""Alien artifact pattern dataset loader."""
from __future__ import annotations

import json
from importlib import resources
from typing import List, Dict, Any

# Path to the dataset JSON file packaged with the library
_DATA_FILE = resources.files("solhunter_zero.data").joinpath("alien_artifact_patterns.json")

_patterns: List[Dict[str, Any]] | None = None


def _load_dataset() -> List[Dict[str, Any]]:
    with _DATA_FILE.open("r", encoding="utf-8") as fh:
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

