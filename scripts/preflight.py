#!/usr/bin/env python3
"""Environment preflight checks for Solsniper-zero."""

from __future__ import annotations

import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Tuple

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


def _parse_dependencies() -> List[str]:
    """Return a list of modules specified in pyproject.toml."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        return []
    if tomllib is None:
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


def check_rust_toolchain() -> Check:
    """Verify that the Rust toolchain is installed."""
    rustc = shutil.which("rustc")  # type: ignore[name-defined]
    cargo = shutil.which("cargo")  # type: ignore[name-defined]
    if not rustc or not cargo:
        return False, "Rust toolchain not found"
    try:
        subprocess.run([rustc, "--version"], check=True, capture_output=True)
    except Exception:
        return False, "rustc failed to execute"
    return True, "Rust toolchain available"


def check_config_file(path: str = "config.toml") -> Check:
    if Path(path).exists():
        return True, f"Found {path}"
    return False, f"Missing {path}"


def check_keypair(dir_path: str = "keypairs") -> Check:
    base = Path(dir_path)
    active = base / "active"
    if not active.exists():
        return False, "keypairs/active not found"
    name = active.read_text().strip()
    keyfile = base / f"{name}.json"
    if not keyfile.exists():
        return False, f"Keypair {keyfile} not found"
    return True, f"Active keypair {name} present"


def check_network(url: str = "https://api.mainnet-beta.solana.com") -> Check:
    try:
        import requests
        requests.get(url, timeout=5)
    except Exception as exc:
        return False, f"Network error: {exc}"
    return True, f"Network access to {url} OK"


def check_gpu() -> Check:
    try:
        from solhunter_zero import device
        try:
            import torch  # type: ignore[import]
        except Exception:  # pragma: no cover - torch is optional
            torch = None  # type: ignore
        if platform.system() == "Darwin" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            return False, "PYTORCH_ENABLE_MPS_FALLBACK=1 not set"
        if not device.detect_gpu():
            return False, "No GPU backend detected"
        if torch is not None and torch.backends.mps.is_available():
            return True, "Metal GPU available"
        return True, "CUDA GPU available"
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)


CHECKS: List[Tuple[str, Callable[[], Check]]] = [
    ("Python", check_python_version),
    ("Dependencies", check_dependencies),
    ("Rust", check_rust_toolchain),
    ("Config", check_config_file),
    ("Keypair", check_keypair),
    ("Network", check_network),
    ("GPU", check_gpu),
]


def main() -> None:
    failed = False
    for name, func in CHECKS:
        ok, msg = func()
        status = "OK" if ok else "FAIL"
        print(f"{name}: {status} - {msg}")
        if not ok:
            failed = True
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
