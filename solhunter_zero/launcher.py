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

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

from .paths import ROOT

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


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
    ``solhunter_zero.macos_setup.prepare_macos_env`` is invoked once to provision
    the interpreter and required toolchain.

    The resolved path is cached in ``.cache/python-exe`` and the
    ``SOLHUNTER_PYTHON`` environment variable. Pass ``--repair`` or set
    ``SOLHUNTER_REPAIR`` to ignore any cached value.
    """

    cache_env = "SOLHUNTER_PYTHON"
    cache_dir = ROOT / ".cache"
    cache_file = cache_dir / "python-exe"

    repair = "--repair" in sys.argv or bool(os.environ.get("SOLHUNTER_REPAIR"))
    if "--repair" in sys.argv:
        sys.argv.remove("--repair")
    if repair:
        try:
            cache_file.unlink()
        except FileNotFoundError:
            pass

    def _finalize(path: str) -> str:
        os.environ[cache_env] = path
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(path)
        except OSError:
            pass
        return path

    if _check_python(sys.executable):
        return _finalize(sys.executable)

    if not repair:
        env_path = os.environ.get(cache_env)
        if env_path and _check_python(env_path):
            return _finalize(env_path)
        if cache_file.exists():
            cached = cache_file.read_text().strip()
            if _check_python(cached):
                return _finalize(cached)
            try:
                cache_file.unlink()
            except FileNotFoundError:
                pass

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
            return _finalize(candidate)

    if platform.system() == "Darwin":
        try:
            from solhunter_zero.macos_setup import prepare_macos_env  # type: ignore
        except Exception:
            prepare_macos_env = None  # type: ignore
        if prepare_macos_env is not None:
            print(
                "Python 3.11 not found; running macOS setup...",
                file=sys.stderr,
            )
            prepare_macos_env(non_interactive=True)
            for name in ("python3.11", "python3", "python"):
                path = shutil.which(name)
                if path and _check_python(path):
                    return _finalize(path)

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


PYTHON_EXE = find_python()
if Path(PYTHON_EXE).resolve() != Path(sys.executable).resolve():
    launcher = Path(__file__).resolve()
    os.execv(PYTHON_EXE, [PYTHON_EXE, str(launcher), *sys.argv[1:]])
    raise SystemExit(1)
from solhunter_zero.macos_setup import ensure_tools  # noqa: E402
from solhunter_zero.bootstrap_utils import ensure_venv  # noqa: E402
from solhunter_zero.logging_utils import log_startup, setup_logging  # noqa: E402

setup_logging("startup")
ensure_tools(non_interactive=True)
ensure_venv(None)
log_startup(f"Virtual environment: {sys.prefix}")

import solhunter_zero.env_config as env_config  # noqa: E402

env_config.configure_environment(ROOT)
from solhunter_zero import device  # noqa: E402

from solhunter_zero.system import set_rayon_threads  # noqa: E402
from solhunter_zero.rpc_utils import ensure_rpc  # noqa: E402


def main(argv: list[str] | None = None) -> NoReturn:
    argv = sys.argv[1:] if argv is None else argv

    # Configure Rayon thread count once for all downstream imports
    set_rayon_threads()
    try:
        device.initialize_gpu()
    except RuntimeError as exc:
        if platform.system() == "Darwin" and platform.machine() == "x86_64":
            print(
                "GPU initialization failed: running under Rosetta. "
                "Re-run using 'arch -arm64' to use the native arm64 Python interpreter.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
        raise

    if "--skip-preflight" not in argv:
        ensure_rpc()

    if "--one-click" not in argv:
        argv.insert(0, "--one-click")
    if "--full-deps" not in argv:
        idx = 1 if argv and argv[0] == "--one-click" else 0
        argv.insert(idx, "--full-deps")

    python_exe = sys.executable
    cmd = [python_exe, "-m", "scripts.startup", *argv]

    if platform.system() == "Darwin":
        cmd = ["arch", "-arm64", *cmd]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()

