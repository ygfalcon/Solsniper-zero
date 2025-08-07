#!/usr/bin/env python3
"""Compatibility wrapper for macOS environment setup."""

from solhunter_zero.macos_setup import *  # noqa: F401,F403

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    from solhunter_zero.macos_setup import main

    main()
