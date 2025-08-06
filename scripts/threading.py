#!/usr/bin/env python3
"""Utility helpers for configuring Rayon threading."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Iterable


def _try_commands(cmds: Iterable[list[str]]) -> int | None:
    """Try each command in order and return the first successful integer output."""
    for cmd in cmds:
        try:
            out = subprocess.check_output(cmd, text=True).strip()
            return int(out)
        except Exception:
            continue
    return None


def _detect_cpu_count() -> int:
    """Detect the number of CPUs across platforms."""
    system = platform.system()
    if system == "Linux":
        count = _try_commands([["nproc"], ["getconf", "_NPROCESSORS_ONLN"]])
    elif system == "Darwin":
        count = _try_commands([["sysctl", "-n", "hw.ncpu"]])
    else:
        count = None
    return count or os.cpu_count() or 1


def print_thread_count() -> None:
    """Print the detected CPU count."""
    print(_detect_cpu_count())


if __name__ == "__main__":  # pragma: no cover
    print_thread_count()
