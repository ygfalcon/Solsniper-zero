#!/usr/bin/env python3
"""Unified launcher for SolHunter Zero.

This script mirrors the behaviour of the previous ``start.command`` shell
script so that all entry points invoke the same Python-based logic.

* Uses ``.venv`` if present.
* Sets ``RAYON_NUM_THREADS`` based on the CPU count.
* Ensures a native arm64 Python on macOS.
* Delegates all arguments to ``scripts/startup.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NoReturn

from .paths import ROOT
from .python_env import find_python

find_python()

sys.path.insert(0, str(ROOT))
from solhunter_zero.macos_setup import ensure_tools  # noqa: E402
from solhunter_zero.bootstrap_utils import ensure_venv  # noqa: E402
from solhunter_zero.logging_utils import log_startup, rotate_startup_log  # noqa: E402

rotate_startup_log()
ensure_tools(non_interactive=True)
ensure_venv(None)
log_startup(f"Virtual environment: {sys.prefix}")

import solhunter_zero.env_config as env_config  # noqa: E402

env_config.configure_environment(ROOT)
from solhunter_zero import device  # noqa: E402
os.chdir(ROOT)

from solhunter_zero.system import set_rayon_threads  # noqa: E402


def main(argv: list[str] | None = None) -> NoReturn:
    argv = sys.argv[1:] if argv is None else argv

    # Configure Rayon thread count once for all downstream imports
    set_rayon_threads()
    device.initialize_gpu()

    if "--one-click" not in argv:
        argv.insert(0, "--one-click")
    if "--full-deps" not in argv:
        idx = 1 if argv and argv[0] == "--one-click" else 0
        argv.insert(idx, "--full-deps")

    python_exe = sys.executable
    startup = ROOT / "scripts" / "startup.py"
    cmd = [python_exe, str(startup), *argv]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()

