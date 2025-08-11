#!/usr/bin/env python3
"""Unified launcher for SolHunter Zero.

This script mirrors the behaviour of the previous ``start.command`` shell
script so that all entry points invoke the same Python-based logic.

* Uses ``.venv`` if present.
* Caches the detected Python interpreter for faster startups.
* Sets ``RAYON_NUM_THREADS`` based on the CPU count.
* On macOS re-execs via ``arch -arm64`` to ensure native arm64 binaries.
* Delegates all arguments to ``scripts/startup.py``.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Callable, NoReturn

from .paths import ROOT
from .python_env import find_python

FAST_MODE = False


def write_ok_marker(path: Path) -> None:
    """Write an ``ok`` marker file, creating parent directories."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok")
    except OSError:
        pass


def _ensure_arm64_python() -> None:
    """Re-exec via ``arch -arm64`` when running under Rosetta on macOS."""
    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        arch = shutil.which("arch")
        if arch:
            try:
                os.execvp(arch, [arch, "-arm64", sys.executable, *sys.argv])
            except OSError as exc:  # pragma: no cover - hard failure
                print(
                    f"Failed to re-exec via 'arch -arm64': {exc}\n"
                    "Please run with an arm64 Python interpreter.",
                    file=sys.stderr,
                )
                raise SystemExit(1)
        else:  # pragma: no cover - hard failure
            print(
                "The 'arch' command was not found; unable to launch arm64 Python.",
                file=sys.stderr,
            )
            raise SystemExit(1)


def configure() -> list[str]:
    """Parse launcher arguments and ensure a suitable interpreter.

    Returns the remaining arguments to forward to ``scripts.startup``.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parsed_args, forward_args = parser.parse_known_args(sys.argv[1:])

    global FAST_MODE
    FAST_MODE = parsed_args.fast or bool(os.environ.get("SOLHUNTER_FAST"))

    python_exe = find_python(repair=parsed_args.repair)
    if Path(python_exe).resolve() != Path(sys.executable).resolve():
        launcher = Path(__file__).resolve()
        try:
            os.execv(python_exe, [python_exe, str(launcher), *forward_args])
        except OSError as exc:  # pragma: no cover - hard failure
            print(
                f"Failed to re-exec launcher via {python_exe}: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    return forward_args


def main(argv: list[str] | None = None) -> NoReturn:
    _ensure_arm64_python()
    forward_args = configure()
    argv = list(forward_args) if argv is None else list(argv)

    from solhunter_zero.macos_setup import ensure_tools  # noqa: E402
    from solhunter_zero.bootstrap_utils import ensure_venv  # noqa: E402
    from solhunter_zero.logging_utils import (  # noqa: E402
        log_startup,
        setup_logging,
    )
    from solhunter_zero.cache_paths import (  # noqa: E402
        TOOLS_OK_MARKER,
        VENV_OK_MARKER,
    )

    setup_logging("startup")

    def _ensure_once(
        marker: Path, action: Callable[[], None], description: str
    ) -> None:
        if FAST_MODE and marker.exists():
            log_startup(f"Fast mode: skipping {description}")
            return
        action()
        if not marker.exists():
            write_ok_marker(marker)

    _ensure_once(
        TOOLS_OK_MARKER,
        lambda: ensure_tools(non_interactive=True),
        "ensure_tools",
    )
    _ensure_once(VENV_OK_MARKER, lambda: ensure_venv(None), "ensure_venv")

    log_startup(f"Virtual environment: {sys.prefix}")

    import solhunter_zero.env_config as env_config  # noqa: E402

    env_config.configure_startup_env(ROOT)
    from solhunter_zero import device  # noqa: E402

    from solhunter_zero.system import set_rayon_threads  # noqa: E402

    # Configure Rayon thread count once for all downstream imports
    set_rayon_threads()
    if not (platform.system() == "Darwin" and platform.machine() == "x86_64"):
        device.initialize_gpu()

    if "--one-click" not in argv:
        argv.insert(0, "--one-click")
    if "--full-deps" not in argv:
        idx = 1 if argv and argv[0] == "--one-click" else 0
        argv.insert(idx, "--full-deps")

    python_exe = sys.executable
    script = "scripts.startup"
    cmd = [python_exe, "-m", script, *argv]

    try:
        os.execvp(cmd[0], cmd)
    except OSError as exc:  # pragma: no cover - hard failure
        print(f"Failed to launch {script}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
