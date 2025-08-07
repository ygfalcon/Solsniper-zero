#!/usr/bin/env python3
"""Environment preflight checks for Solsniper-zero."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Tuple
from urllib import error
import json

from scripts.deps import check_deps
from solhunter_zero.config_utils import (
    ensure_default_config,
    select_active_keypair,
)
from solhunter_zero import wallet



ROOT = Path(__file__).resolve().parent.parent

Check = Tuple[bool, str]


def check_python_version(min_version: tuple[int, int] = (3, 11)) -> Check:
    """Ensure the running Python interpreter meets the minimum version."""
    if sys.version_info < min_version:
        return False, f"Python {min_version[0]}.{min_version[1]}+ required"
    return True, (
        f"Python {sys.version_info.major}.{sys.version_info.minor} detected"
    )


def check_dependencies() -> Check:
    """Report any missing required or optional dependencies."""
    missing_required, missing_optional = check_deps()
    if missing_required or missing_optional:
        parts: List[str] = []
        if missing_required:
            parts.append(
                "missing required: " + ", ".join(sorted(missing_required))
            )
        if missing_optional:
            parts.append(
                "missing optional: " + ", ".join(sorted(missing_optional))
            )
        return False, "; ".join(parts)
    return True, "All dependencies available"


def check_homebrew() -> Check:
    """Verify that the Homebrew package manager is installed."""
    brew = shutil.which("brew")  # type: ignore[name-defined]
    if not brew:
        return False, "Homebrew not found"
    try:
        subprocess.run([brew, "--version"], check=True, capture_output=True)
    except Exception:
        return False, "brew failed to execute"
    return True, "Homebrew available"


def check_rustup() -> Check:
    """Verify that rustup is installed."""
    rustup = shutil.which("rustup")  # type: ignore[name-defined]
    if not rustup:
        return False, "rustup not found"
    try:
        subprocess.run([rustup, "--version"], check=True, capture_output=True)
    except Exception:
        return False, "rustup failed to execute"
    return True, "rustup available"


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


def check_xcode_clt() -> Check:
    """Verify that the Xcode Command Line Tools are installed on macOS."""
    if sys.platform != "darwin":
        return True, "Not macOS"
    try:
        subprocess.run(["xcode-select", "-p"], check=True, capture_output=True)
    except FileNotFoundError:
        return False, "xcode-select not found"
    except subprocess.CalledProcessError:
        return False, "Xcode Command Line Tools not installed"
    return True, "Xcode Command Line Tools available"


def check_config_file(path: str = "config.toml") -> Check:
    cfg = ensure_default_config()
    if cfg.exists():
        return True, f"Found {cfg}"
    return False, f"Missing {path}"


def check_keypair(dir_path: str = "keypairs") -> Check:
    try:
        info = select_active_keypair(auto=True)
    except Exception:
        return False, "keypairs/active not found"
    keyfile = Path(wallet.KEYPAIR_DIR) / f"{info.name}.json"
    if keyfile.exists():
        return True, f"Active keypair {info.name} present"
    return False, f"Keypair {keyfile} not found"


def check_required_env(keys: List[str] | None = None) -> Check:
    """Ensure critical environment variables are configured.

    Parameters
    ----------
    keys:
        Optional list of variable names to verify. Defaults to
        ``["SOLANA_RPC_URL", "BIRDEYE_API_KEY"]``.
    """

    required = keys or ["SOLANA_RPC_URL", "BIRDEYE_API_KEY"]
    missing = []
    for key in required:
        val = os.getenv(key)
        if not val or val in {"", "YOUR_BIRDEYE_KEY", "YOUR_BIRDEYE_API_KEY"}:
            missing.append(key)
    if missing:
        joined = ", ".join(missing)
        return False, f"Missing environment variables: {joined}. Set them and retry"
    return True, "Required environment variables set"


def check_network(default_url: str = "https://api.mainnet-beta.solana.com") -> Check:
    url = os.environ.get("SOLANA_RPC_URL", default_url)
    try:
        from solhunter_zero.http import check_endpoint

        check_endpoint(url)
    except error.URLError as exc:
        return False, f"Network error: {exc}"
    return True, f"Network access to {url} OK"


def check_gpu() -> Check:
    """Report GPU availability and the selected default device."""
    try:
        from solhunter_zero import device

        return device.verify_gpu()
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)


CHECKS: List[Tuple[str, Callable[[], Check]]] = [
    ("Python", check_python_version),
    ("Dependencies", check_dependencies),
    ("Homebrew", check_homebrew),
    ("Rustup", check_rustup),
    ("Rust", check_rust_toolchain),
    ("Xcode CLT", check_xcode_clt),
    ("Config", check_config_file),
    ("Keypair", check_keypair),
    ("Environment", check_required_env),
    (
        "Network",
        lambda: check_network(
            os.environ.get(
                "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
            )
        ),
    ),
    ("GPU", check_gpu),
]


def run_preflight() -> List[Tuple[str, bool, str]]:
    """Run all preflight checks and return their results.

    The return value is a list of tuples containing the check name, a boolean
    indicating success, and a descriptive message.
    """

    results: List[Tuple[str, bool, str]] = []
    for name, func in CHECKS:
        ok, msg = func()
        results.append((name, ok, msg))

    data = {
        "successes": [
            {"name": name, "message": msg} for name, ok, msg in results if ok
        ],
        "failures": [
            {"name": name, "message": msg} for name, ok, msg in results if not ok
        ],
    }
    try:
        with open(ROOT / "preflight.json", "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except OSError:
        pass

    return results


def main() -> None:
    failures: List[Tuple[str, str]] = []
    results = run_preflight()
    for name, ok, msg in results:
        status = "OK" if ok else "FAIL"
        print(f"{name}: {status} - {msg}")
        if not ok:
            failures.append((name, msg))

    if failures:
        print("\nSummary of failures:")
        for name, msg in failures:
            print(f"- {name}: {msg}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
