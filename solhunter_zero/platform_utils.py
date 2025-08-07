"""Cross-platform utilities for system and architecture checks.

This module centralizes detection of the current operating system and CPU
architecture to avoid sprinkling ``platform.system`` and ``platform.machine``
checks across the codebase.  Helper predicates provide meaningful names for
common combinations used throughout the project.
"""

from __future__ import annotations

import platform


def system() -> str:
    """Return the current operating system name.

    This is a thin wrapper around :func:`platform.system` which allows tests to
    monkeypatch the value without touching the global :mod:`platform` module.
    """

    return platform.system()


def machine() -> str:
    """Return the machine architecture string.

    The result typically resembles ``"x86_64"`` or ``"arm64"``.
    """

    return platform.machine()


def is_macos() -> bool:
    """Return ``True`` if running on macOS."""

    return system() == "Darwin"


def is_macos_arm64() -> bool:
    """Return ``True`` when running on macOS with Apple Silicon."""

    return is_macos() and machine() == "arm64"


def requires_rosetta() -> bool:
    """Return ``True`` when executing under Rosetta translation.

    This indicates a macOS environment running an ``x86_64`` binary, typically
    via Rosetta on Apple Silicon.  In this mode native Metal acceleration is
    unavailable.
    """

    return is_macos() and machine() == "x86_64"

