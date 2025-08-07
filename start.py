#!/usr/bin/env python3
"""Shim that forwards execution to ``solhunter_zero.startup``."""

import os
import sys


if __name__ == "__main__":
    os.execv(
        sys.executable,
        [sys.executable, "-m", "solhunter_zero.startup", *sys.argv[1:]],
    )
