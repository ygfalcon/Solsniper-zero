#!/usr/bin/env python3
"""Deprecated shim for backward compatibility.

This module re-exports :func:`solhunter_zero.launcher.main`.
"""
from __future__ import annotations

import platform
import sys
import warnings

if platform.machine() != "arm64":
    print(
        "Non-arm64 Python detected. Please install an arm64 build of Python to run SolHunter Zero.",
    )
    raise SystemExit(1)

from solhunter_zero.launcher import main

warnings.warn(
    "scripts.launcher is deprecated; use solhunter_zero.launcher instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - legacy behaviour
    main()
