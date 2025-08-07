#!/usr/bin/env python3
"""Deprecated shim for backward compatibility.

This module re-exports :func:`solhunter_zero.launcher.main`.
"""
from __future__ import annotations

from solhunter_zero.macos_setup import ensure_arm64

ensure_arm64()

import warnings

from solhunter_zero.launcher import main

warnings.warn(
    "scripts.launcher is deprecated; use solhunter_zero.launcher instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - legacy behaviour
    main()
