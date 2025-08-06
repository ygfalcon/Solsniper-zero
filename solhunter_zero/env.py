from __future__ import annotations

"""Utilities for loading environment variables from files."""

from pathlib import Path
import os

__all__ = ["load_env_file"]


def load_env_file(path: Path) -> None:
    """Load ``KEY=VALUE`` pairs from *path* into ``os.environ``.

    Blank lines and ``#`` comments are ignored. Existing environment variables
    are preserved. Missing files are silently ignored.
    """

    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)
