#!/usr/bin/env python3
"""Perform a one-click setup and launch for SolHunter Zero."""

from __future__ import annotations

import importlib.resources as resources
import os
import sys
import subprocess
import shutil

from solhunter_zero.macos_setup import ensure_tools
import solhunter_zero.env_config as env_config
from solhunter_zero.paths import ROOT
from scripts import quick_setup
from solhunter_zero.bootstrap_utils import ensure_deps
from solhunter_zero import device
from solhunter_zero.logging_utils import log_startup


def main(argv: list[str] | None = None) -> None:
    """Execute the automated setup steps then run the autopilot."""
    ensure_tools(non_interactive=True)
    env_config.configure_environment(ROOT)
    quick_setup.main(["--auto", "--non-interactive"])
    ensure_deps(install_optional=True)

    if shutil.which("cargo") and shutil.which("rustup"):
        try:
            subprocess.check_call(
                [
                    "cargo",
                    "build",
                    "--release",
                    "--features=parallel",
                    "-p",
                    "route_ffi",
                ],
                cwd=ROOT,
            )
        except subprocess.CalledProcessError as exc:
            msg = "Failed to build route_ffi with parallel feature"
            print(f"{msg}: {exc}")
            log_startup(f"{msg}: {exc}")

    device.initialize_gpu()

    start_all = resources.files("scripts") / "start_all.py"
    cmd = [sys.executable, str(start_all), "autopilot"]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":  # pragma: no cover
    main()
