#!/usr/bin/env python3
"""Shim that forwards execution to ``scripts/one_click.py``."""

import os
import sys
from pathlib import Path


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "one_click.py"
    os.execv(sys.executable, [sys.executable, str(script), *sys.argv[1:]])
