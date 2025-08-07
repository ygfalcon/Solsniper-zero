#!/usr/bin/env python3
"""Unified launcher for SolHunter Zero.

This script mirrors the behaviour of the previous ``start.command`` shell
script so that all entry points invoke the same Python-based logic.

* Uses ``.venv`` if present.
* Sets ``RAYON_NUM_THREADS`` based on the CPU count.
* On macOS re-execs via ``arch -arm64`` to ensure native arm64 binaries.
* Delegates all arguments to ``scripts/startup.py``.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


ROOT = Path(__file__).resolve().parent.parent


def _check_python(exe: str) -> bool:
    """Return ``True`` if ``exe`` is a Python >=3.11 interpreter."""
    try:
        out = subprocess.check_output(
            [exe, "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"],
            text=True,
        ).strip()
        major, minor = map(int, out.split(".")[:2])
        return (major, minor) >= (3, 11)
    except Exception:  # pragma: no cover - defensive
        return False


def find_python() -> str:
    """Locate a suitable Python 3.11 interpreter.

    If the current interpreter is already adequate, it is returned. Otherwise
    search common locations including ``.venv`` and system ``PATH``. On macOS
    ``scripts.mac_setup.auto_prepare_macos`` is invoked once to provision the
    interpreter if necessary.
    """

    if _check_python(sys.executable):
        return sys.executable

    candidates: list[str] = []

    # Existing virtual environment interpreters
    venv = ROOT / ".venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    for name in ("python3.11", "python3", "python"):
        p = bin_dir / name
        if p.exists():
            candidates.append(str(p))

    # Interpreters on PATH
    for name in ("python3.11", "python3", "python"):
        path = shutil.which(name)
        if path:
            candidates.append(path)

    for candidate in candidates:
        if _check_python(candidate):
            return candidate

    if platform.system() == "Darwin":
        try:
            from . import mac_setup
        except Exception:  # pragma: no cover - defensive
            mac_setup = None
        if mac_setup is not None:
            print(
                "Python 3.11 not found; running macOS setup...",
                file=sys.stderr,
            )
            mac_setup.auto_prepare_macos(non_interactive=True)
            for name in ("python3.11", "python3", "python"):
                path = shutil.which(name)
                if path and _check_python(path):
                    return path

    message = "Python 3.11 or higher is required."
    if platform.system() == "Darwin":
        message += " Run 'scripts/mac_setup.py --non-interactive' to install Python 3.11."
    else:
        message += " Please install Python 3.11 and try again."
    print(message, file=sys.stderr)
    raise SystemExit(1)


PYTHON_EXE = find_python()
if Path(PYTHON_EXE).resolve() != Path(sys.executable).resolve():
    launcher = Path(__file__).resolve()
    os.execv(PYTHON_EXE, [PYTHON_EXE, str(launcher), *sys.argv[1:]])
    raise SystemExit(1)


sys.path.insert(0, str(ROOT))
from solhunter_zero.bootstrap_utils import ensure_venv  # noqa: E402
from solhunter_zero.logging_utils import log_startup  # noqa: E402

ensure_venv(None)
log_startup(f"Virtual environment: {sys.prefix}")

from solhunter_zero import env  # noqa: E402

env.load_env_file(ROOT / ".env")
from solhunter_zero import device  # noqa: E402
os.chdir(ROOT)

from solhunter_zero.system import set_rayon_threads  # noqa: E402


def main(argv: list[str] | None = None) -> NoReturn:
    argv = sys.argv[1:] if argv is None else argv

    # Configure Rayon thread count once for all downstream imports
    set_rayon_threads()
    device.ensure_gpu_env()
    with open(ROOT / "startup.log", "a", encoding="utf-8") as fh:
        fh.write(f"TORCH_DEVICE: {os.environ.get('TORCH_DEVICE', 'cpu')}\n")

    if "--one-click" not in argv:
        argv.insert(0, "--one-click")
    if "--full-deps" not in argv:
        idx = 1 if argv and argv[0] == "--one-click" else 0
        argv.insert(idx, "--full-deps")

    python_exe = sys.executable
    startup = ROOT / "scripts" / "startup.py"
    cmd = [python_exe, str(startup), *argv]

    if platform.system() == "Darwin":
        cmd = ["arch", "-arm64", *cmd]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()

