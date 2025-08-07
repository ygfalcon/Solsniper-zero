#!/usr/bin/env python3
"""Deprecated shim for backward compatibility.

This module re-exports :func:`solhunter_zero.launcher.main`.
"""
from __future__ import annotations

import warnings

from solhunter_zero.launcher import main as _main
from solhunter_zero.startup_utils import bootstrap_all

warnings.warn(
    "scripts.launcher is deprecated; use solhunter_zero.launcher instead",
    DeprecationWarning,
    stacklevel=2,
)

def main() -> None:
    bootstrap_all()
    _main()


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - legacy behaviour
    main()
