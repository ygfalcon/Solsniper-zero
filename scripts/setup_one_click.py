#!/usr/bin/env python3
"""Perform a one-click setup and launch for SolHunter Zero."""

from __future__ import annotations

import os
import sys

from solhunter_zero.macos_setup import ensure_tools
import solhunter_zero.env_config as env_config
from solhunter_zero.paths import ROOT
from scripts import quick_setup
from solhunter_zero.bootstrap_utils import ensure_deps
from solhunter_zero import device


def main(argv: list[str] | None = None) -> None:
    """Execute the automated setup steps then run the autopilot."""
    ensure_tools(non_interactive=True)
    env_config.configure_environment(ROOT)
    quick_setup.main(["--auto", "--non-interactive"])
    ensure_deps(install_optional=True)
    device.initialize_gpu()

    os.chdir(ROOT)
    start_all = ROOT / "scripts" / "start_all.py"
    cmd = [sys.executable, str(start_all), "autopilot"]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":  # pragma: no cover
    main()
