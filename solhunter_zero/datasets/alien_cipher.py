"""Loader for alien cipher coefficients dataset."""
from __future__ import annotations

import json
from importlib import resources
from typing import Any, Dict

# Default dataset path packaged with the library
_DEFAULT_PATH = resources.files("solhunter_zero.data").joinpath("alien_cipher.json")

# Module level cache so we do not repeatedly load the same file
_cache_path: str | None = None
_cache_data: Dict[str, Any] | None = None


def load_alien_cipher(path: str = _DEFAULT_PATH) -> Dict[str, Any]:
    """Return the alien cipher coefficient mapping located at ``path``.

    The result is cached on the module level; subsequent calls with the same
    ``path`` will reuse the previously loaded data.
    """
    global _cache_path, _cache_data
    if _cache_data is not None and _cache_path == path:
        return _cache_data

    try:
        if hasattr(path, "open"):
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
    except Exception:
        _cache_path = path
        _cache_data = {}
        return _cache_data

    if not isinstance(data, dict):
        data = {}

    _cache_path = path
    _cache_data = data
    return data
