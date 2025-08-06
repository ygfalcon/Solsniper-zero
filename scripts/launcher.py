#!/usr/bin/env python3
"""Unified launcher for SolHunter Zero.

The launcher ensures the runtime environment is prepared before delegating to
``solhunter_zero.main``.  It mirrors the behaviour of the previous shell-based
entry points and centralises environment preparation via
``solhunter_zero.startup.prepare_environment``.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

from solhunter_zero.startup import prepare_environment

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def _cpu_count(python: str) -> int:
    try:
        out = subprocess.check_output(
            [python, "-m", "solhunter_zero.system", "cpu-count"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return int(out)
    except Exception:
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.ncpu"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return int(out)
        except Exception:
            return 1


def main(argv: list[str] | None = None) -> "NoReturn":
    argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(description="SolHunter Zero launcher")
    parser.add_argument("--one-click", action="store_true", help="non-interactive startup")
    parser.add_argument("--full-deps", action="store_true", help="install optional dependencies")
    args, rest = parser.parse_known_args(argv)

    if args.full_deps:
        os.environ["SOLHUNTER_INSTALL_OPTIONAL"] = "1"

    # Re-exec under native arm64 on macOS if necessary before preparing the env
    if platform.system() == "Darwin" and os.getenv("SOLHUNTER_NO_ARCH") != "1":
        threads = _cpu_count(sys.executable)
        os.environ["RAYON_NUM_THREADS"] = str(threads)
        os.environ["SOLHUNTER_NO_ARCH"] = "1"
        cmd = ["arch", "-arm64", sys.executable, __file__, *argv]
        os.execvp(cmd[0], cmd)

    prepare_environment(one_click=args.one_click)

    cmd = [sys.executable, "-m", "solhunter_zero.main", *rest]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
