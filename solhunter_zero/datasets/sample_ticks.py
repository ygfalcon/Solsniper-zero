"""Loader for sample tick history used in tests and demos."""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List

# Path to the sample ticks dataset bundled with the repository
DEFAULT_PATH = resources.files(__package__).joinpath(
    "..",
    "..",
    "datasets",
    "sample_ticks.json",
)

# Backwards compatibility
_DEFAULT_PATH = DEFAULT_PATH

_cache_path: str | None = None
_cache_data: List[Dict[str, Any]] | None = None


def load_sample_ticks(path: Path | str = DEFAULT_PATH) -> List[Dict[str, Any]]:
    """Return sample tick entries located at ``path``."""
    global _cache_path, _cache_data
    if _cache_data is not None and _cache_path == str(path):
        return _cache_data

    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        _cache_path = str(path)
        _cache_data = []
        return _cache_data

    if not isinstance(data, list):
        data = []

    _cache_path = str(path)
    _cache_data = data
    return data
