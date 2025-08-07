#!/usr/bin/env python3
"""Run Solsniper Zero with minimal user interaction."""

from __future__ import annotations

import platform
import sys

from solhunter_zero.bootstrap_utils import ensure_venv


def main(argv: list[str] | None = None) -> None:
    """Execute preflight checks then launch the existing startup routine."""
    ensure_venv(None)
    if platform.system() == "Darwin":
        from scripts import mac_env
        mac_env.prepare_macos_env()
    from scripts import preflight, launcher
    preflight.main()
    launcher.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
