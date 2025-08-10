#!/usr/bin/env python3
"""Canonical entry point for SolHunter Zero.

The script resolves the repository root, switches to it as the working
directory and prepends it to ``sys.path`` before delegating to
``solhunter_zero.launcher.main``. This allows ``start.py`` to be invoked from
any directory.
"""

from pathlib import Path
import os
import sys

# Resolve the repository root and ensure it's the working directory. Prepend
# the path to ``sys.path`` so imports succeed regardless of invocation
# location.
REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from solhunter_zero.launcher import main


if __name__ == "__main__":
    main()
