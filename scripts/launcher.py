#!/usr/bin/env python3
"""Unified launcher for SolHunter Zero.

This script mirrors the behaviour of the previous ``start.command`` shell
script so that all entry points invoke the same Python-based logic.

* Uses ``.venv`` if present.
* On macOS, sets ``RAYON_NUM_THREADS`` based on the CPU count and re-execs
  via ``arch -arm64`` to ensure native arm64 binaries.
* Delegates all arguments to ``scripts/startup.py``.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from solhunter_zero.system import detect_cpu_count  # noqa: E402


def main(argv: list[str] | None = None) -> NoReturn:
    argv = sys.argv[1:] if argv is None else argv

    python_exe = sys.executable
    venv = ROOT / ".venv"
    for candidate in (venv / "bin" / "python3", venv / "bin" / "python"):
        if candidate.exists():
            python_exe = str(candidate)
            break

    startup = ROOT / "scripts" / "startup.py"
    cmd = [python_exe, str(startup), *argv]

    if platform.system() == "Darwin":
        os.environ["RAYON_NUM_THREADS"] = str(detect_cpu_count())
        cmd = ["arch", "-arm64", *cmd]
        os.execvp(cmd[0], cmd)
    else:
        os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
