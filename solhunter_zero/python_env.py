#!/usr/bin/env python3
"""Utilities for locating a suitable Python interpreter.

This module centralizes discovery of a Python 3.11+ interpreter and
re-executes the current process if a better interpreter is found.  On macOS it
ensures the interpreter runs natively on arm64.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .paths import ROOT


def _check_python(exe: str) -> bool:
    """Return ``True`` if ``exe`` is Python >=3.11 (arm64 on macOS)."""
    try:
        out = subprocess.check_output(
            [
                exe,
                "-c",
                "import sys, platform; print('.'.join(map(str, sys.version_info[:2])), platform.machine())",
            ],
            text=True,
        ).strip()
        version, machine = out.split()
        major, minor = map(int, version.split(".")[:2])
        if (major, minor) < (3, 11):
            return False
        if platform.system() == "Darwin" and machine != "arm64":
            return False
        return True
    except Exception:  # pragma: no cover - defensive
        return False


def find_python() -> str:
    """Locate a suitable Python interpreter and re-exec if necessary.

    Returns the absolute path to the interpreter.  If the current interpreter is
    unsuitable, the process is re-executed with a better one and this function
    does not return.
    """
    current = str(Path(sys.executable).resolve())
    if _check_python(current):
        return current

    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        cmd = ["arch", "-arm64", current, *sys.argv]
        try:
            os.execvp(cmd[0], cmd)
        except OSError:
            pass  # fall back to searching other interpreters

    candidates: list[str] = []

    venv = ROOT / ".venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    for name in ("python3.11", "python3", "python"):
        p = bin_dir / name
        if p.exists():
            candidates.append(str(p))

    for name in ("python3.11", "python3", "python"):
        path = shutil.which(name)
        if path:
            candidates.append(path)

    def _search(paths: list[str]) -> str | None:
        for cand in paths:
            cand_resolved = str(Path(cand).resolve())
            if _check_python(cand_resolved):
                if cand_resolved != current:
                    if platform.system() == "Darwin":
                        cmd = ["arch", "-arm64", cand_resolved, *sys.argv]
                        os.execvp(cmd[0], cmd)
                    else:
                        os.execv(cand_resolved, [cand_resolved, *sys.argv])
                    raise SystemExit(1)
                return cand_resolved
        return None

    found = _search(candidates)
    if found:
        return found

    if platform.system() == "Darwin":
        try:
            from .macos_setup import prepare_macos_env  # type: ignore
        except Exception:  # pragma: no cover - fallback when module unavailable
            prepare_macos_env = None  # type: ignore
        if prepare_macos_env is not None:
            print(
                "Python 3.11 not found; running macOS setup...",
                file=sys.stderr,
            )
            prepare_macos_env(non_interactive=True)
            new_candidates: list[str] = []
            for name in ("python3.11", "python3", "python"):
                path = shutil.which(name)
                if path:
                    new_candidates.append(path)
            found = _search(new_candidates)
            if found:
                return found

    message = "Python 3.11 or higher is required."
    if platform.system() == "Darwin":
        message += (
            " Run 'python -c \"from solhunter_zero.macos_setup import prepare_macos_env; "
            "prepare_macos_env()\"' to install Python 3.11."
        )
    else:
        message += " Please install Python 3.11 and try again."
    print(message, file=sys.stderr)
    raise SystemExit(1)
