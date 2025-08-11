from __future__ import annotations

"""Context manager for temporary environment flags.

``env_flags`` allows setting one or more environment variables for the
scope of a ``with`` block and automatically restores their previous
values on exit.  ``FLAGS`` lists the available flags used throughout the
project so they are documented in a single location for easier
maintenance.
"""

from contextlib import contextmanager
import os
from typing import Iterator

FLAGS: dict[str, str] = {
    "SOLHUNTER_SKIP_DEPS": "Skip dependency installation.",
    "SOLHUNTER_SKIP_SETUP": "Skip configuration and wallet setup.",
    "SOLHUNTER_SKIP_PREFLIGHT": "Skip preflight checks.",
    "SOLHUNTER_NO_DIAGNOSTICS": "Disable post-run diagnostics collection.",
    "SOLHUNTER_SKIP_VENV": "Skip virtual environment creation.",
    "SOLHUNTER_FAST": "Skip repeated environment checks when markers exist.",
}

__all__ = ["env_flags", "FLAGS"]


@contextmanager
def env_flags(**flags: str) -> Iterator[None]:
    """Temporarily set environment variables and restore previous values.

    Parameters
    ----------
    **flags:
        Mapping of environment variable names to the values they should be set
        to for the duration of the context manager.
    """

    previous: dict[str, str | None] = {}
    try:
        for key, value in flags.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
