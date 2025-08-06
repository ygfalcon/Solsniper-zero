#!/usr/bin/env python3
"""Cross-platform entry point for SolHunter Zero.

This script replaces the former ``run.sh`` and ``start.command`` wrappers
by delegating directly to :func:`scripts.launcher.main`.
"""
from scripts import launcher

if __name__ == "__main__":
    launcher.main()
