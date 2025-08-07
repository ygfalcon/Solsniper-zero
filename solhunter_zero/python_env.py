from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

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


def find_python(*, reexec: bool = False) -> str:
    """Locate a suitable Python 3.11 interpreter.

    Parameters
    ----------
    reexec:
        When ``True`` and a different interpreter is found, re-exec the current
        process under that interpreter.
    """
    if _check_python(sys.executable):
        python = sys.executable
    else:
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
        python = None
        for candidate in candidates:
            if _check_python(candidate):
                python = candidate
                break
        if not python and platform.system() == "Darwin":
            setup = ROOT / "scripts" / "mac_setup.py"
            if setup.exists():
                print(
                    "Python 3.11 not found; running macOS setup...",
                    file=sys.stderr,
                )
                subprocess.run(
                    [sys.executable, str(setup), "--non-interactive"],
                    check=False,
                )
                for name in ("python3.11", "python3", "python"):
                    path = shutil.which(name)
                    if path and _check_python(path):
                        python = path
                        break
        if not python:
            message = "Python 3.11 or higher is required."
            if platform.system() == "Darwin":
                message += " Run 'scripts/mac_setup.py --non-interactive' to install Python 3.11."
            else:
                message += " Please install Python 3.11 and try again."
            print(message, file=sys.stderr)
            raise SystemExit(1)
    if reexec and Path(python).resolve() != Path(sys.executable).resolve():
        script = Path(sys.argv[0]).resolve()
        os.execv(python, [python, str(script), *sys.argv[1:]])
        raise SystemExit(1)
    return python
