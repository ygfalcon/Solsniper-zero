#!/usr/bin/env python3
"""Common environment checks for Solsniper-zero."""

from __future__ import annotations

import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - only for very old Pythons
    tomllib = None

Check = Tuple[bool, str]


def check_python_version(min_version: tuple[int, int] = (3, 11)) -> Check:
    """Ensure the running Python interpreter meets the minimum version."""
    if sys.version_info < min_version:
        return False, f"Python {min_version[0]}.{min_version[1]}+ required"
    return True, f"Python {sys.version_info.major}.{sys.version_info.minor} detected"


def _parse_dependencies() -> Iterable[str]:
    """Return a list of modules specified in pyproject.toml."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists() or tomllib is None:
        return []
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    deps: List[str] = []
    for entry in data.get("project", {}).get("dependencies", []):
        name = re.split(r"[<>=\[]", entry, 1)[0].strip()
        deps.append(name)
    return deps


def check_dependencies() -> Check:
    """Attempt to import each dependency declared in pyproject.toml."""
    missing: List[str] = []
    for dep in _parse_dependencies():
        module_name = dep.replace("-", "_")
        if module_name == "scikit_learn":  # special case
            module_name = "sklearn"
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(dep)
    if missing:
        return False, f"Missing modules: {', '.join(sorted(missing))}"
    return True, "All dependencies available"


def check_homebrew(
    which: Callable[[str], str | None] = shutil.which,
) -> Check:
    """Verify that the Homebrew package manager is installed."""
    brew = which("brew")  # type: ignore[name-defined]
    if not brew:
        return False, "Homebrew not found"
    try:
        subprocess.run([brew, "--version"], check=True, capture_output=True)
    except Exception:
        return False, "brew failed to execute"
    return True, "Homebrew available"


def check_rustup(
    which: Callable[[str], str | None] = shutil.which,
) -> Check:
    """Verify that rustup is installed."""
    rustup = which("rustup")  # type: ignore[name-defined]
    if not rustup:
        return False, "rustup not found"
    try:
        subprocess.run([rustup, "--version"], check=True, capture_output=True)
    except Exception:
        return False, "rustup failed to execute"
    return True, "rustup available"


def check_rust_toolchain(
    which: Callable[[str], str | None] = shutil.which,
) -> Check:
    """Verify that the Rust toolchain is installed."""
    rustc = which("rustc")  # type: ignore[name-defined]
    cargo = which("cargo")  # type: ignore[name-defined]
    if not rustc or not cargo:
        return False, "Rust toolchain not found"
    try:
        subprocess.run([rustc, "--version"], check=True, capture_output=True)
    except Exception:
        return False, "rustc failed to execute"
    return True, "Rust toolchain available"


__all__ = [
    "Check",
    "check_python_version",
    "check_dependencies",
    "check_homebrew",
    "check_rustup",
    "check_rust_toolchain",
]
