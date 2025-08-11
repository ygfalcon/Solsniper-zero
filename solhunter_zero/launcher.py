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
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

from .paths import ROOT

FAST_MODE = False

# In-memory cache for the resolved interpreter path
_PYTHON_CACHE: str | None = None


def write_ok_marker(path: Path) -> None:
    """Write an ``ok`` marker file, creating parent directories."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok")
    except OSError:
        pass


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


def find_python(repair: bool = False) -> str:
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

    global _PYTHON_CACHE
    repair = repair or bool(os.environ.get("SOLHUNTER_REPAIR"))

    if repair:
        _PYTHON_CACHE = None
        try:
            cache_file.unlink()
        except FileNotFoundError:
            pass
    else:
        if _PYTHON_CACHE is not None:
            return _PYTHON_CACHE
        env_path = os.environ.get(cache_env)
        if env_path and _check_python(env_path):
            _PYTHON_CACHE = env_path
            return env_path

    def _finalize(path: str) -> str:
        os.environ[cache_env] = path
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(path)
        except OSError:
            pass
        _PYTHON_CACHE = path
        return path

    if _check_python(sys.executable):
        return _finalize(sys.executable)

    if not repair and cache_file.exists():
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
    forward_args = configure()
    argv = list(forward_args) if argv is None else list(argv)

    from solhunter_zero.macos_setup import ensure_tools  # noqa: E402
    from solhunter_zero.bootstrap_utils import ensure_venv  # noqa: E402
    from solhunter_zero.logging_utils import log_startup, setup_logging  # noqa: E402
    from solhunter_zero.cache_paths import TOOLS_OK_MARKER, VENV_OK_MARKER  # noqa: E402

    setup_logging("startup")
    if FAST_MODE and TOOLS_OK_MARKER.exists():
        log_startup("Fast mode: skipping ensure_tools")
    else:
        ensure_tools(non_interactive=True)
        if not TOOLS_OK_MARKER.exists():
            write_ok_marker(TOOLS_OK_MARKER)

    if FAST_MODE and VENV_OK_MARKER.exists():
        log_startup("Fast mode: skipping ensure_venv")
    else:
        ensure_venv(None)
        write_ok_marker(VENV_OK_MARKER)

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

