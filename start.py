#!/usr/bin/env python3
"""Canonical entry point for SolHunter Zero.

This script simply delegates to :func:`solhunter_zero.launcher.main`.
"""

from solhunter_zero.launcher import main
from solhunter_zero.logging_utils import startup_logger


if __name__ == "__main__":
    startup_logger()
    main()
