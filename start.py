#!/usr/bin/env python3
"""Shim that forwards execution to ``scripts/launcher.py``."""

import os
import sys
from pathlib import Path


if __name__ == "__main__":
    launcher = Path(__file__).resolve().parent / "scripts" / "launcher.py"
    os.execv(sys.executable, [sys.executable, str(launcher), *sys.argv[1:]])
