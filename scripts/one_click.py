#!/usr/bin/env python3
"""Compatibility wrapper that prepares macOS env then forwards to ``launcher.py``."""

import os
import platform
import sys
from pathlib import Path


if __name__ == "__main__":
    if platform.system() == "Darwin":
        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root))
        try:
            from solhunter_zero.macos_setup import prepare_macos_env

            prepare_macos_env(non_interactive=True)
        except Exception:
            pass
    script = Path(__file__).resolve().parent / "launcher.py"
    os.execv(sys.executable, [sys.executable, str(script), *sys.argv[1:]])
