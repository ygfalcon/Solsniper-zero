"""Shared preflight check utilities for SolHunter Zero.

This module centralizes the various environment checks used by the CLI
scripts.  Functions here are imported by both ``scripts/preflight.py`` and
``scripts/startup.py`` to ensure consistent behaviour regardless of which
entrypoint is executed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
from urllib import error

from scripts.deps import check_deps
from solhunter_zero.config_utils import ensure_default_config, select_active_keypair
from solhunter_zero import wallet
from solhunter_zero.paths import ROOT

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


def check_disk_space(min_bytes: int) -> None:
    """Ensure there is at least ``min_bytes`` free on the current filesystem."""

    try:
        _, _, free = shutil.disk_usage(ROOT)
    except OSError as exc:  # pragma: no cover - unexpected failure
        print(f"Unable to determine free disk space: {exc}")
        raise SystemExit(1)

    if free < min_bytes:
        required_gb = min_bytes / (1024 ** 3)
        free_gb = free / (1024 ** 3)
        print(
            f"Insufficient disk space: {free_gb:.2f} GB available,"
            f" {required_gb:.2f} GB required."
        )
        print("Please free up disk space and try again.")
        raise SystemExit(1)


def check_internet(url: str | None = None) -> None:
    """Ensure basic internet connectivity by reaching a known Solana RPC host.

    Parameters
    ----------
    url:
        Optional URL to test. When ``None`` the function uses the value of
        ``SOLANA_RPC_URL`` from the environment, falling back to the public
        mainnet endpoint if unset.
    """

    import urllib.request
    import time

    target = url or os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

    for attempt in range(3):
        try:
            with urllib.request.urlopen(target, timeout=5) as resp:  # nosec B310
                resp.read()
                return
        except Exception as exc:  # pragma: no cover - network failure
            if attempt == 2:
                print(
                    f"Failed to reach {target} after 3 attempts: {exc}. "
                    "Check your internet connection."
                )
                raise SystemExit(1)
            wait = 2**attempt
            print(
                f"Attempt {attempt + 1} failed to reach {target}: {exc}. "
                f"Retrying in {wait} seconds..."
            )
            time.sleep(wait)


def check_required_env(keys: List[str] | None = None) -> Check:
    """Ensure critical environment variables are configured."""

    required = keys or ["SOLANA_RPC_URL", "BIRDEYE_API_KEY"]
    missing = []
    for key in required:
        val = os.getenv(key)
        if not val or val in {"", "YOUR_BIRDEYE_KEY", "YOUR_BIRDEYE_API_KEY"}:
            missing.append(key)
    if missing:
        joined = ", ".join(missing)
        return False, (
            f"Missing environment variables: {joined}. "
            "Set them and retry"
        )
    return True, "Required environment variables set"


def check_network(default_url: str = "https://api.mainnet-beta.solana.com") -> Check:
    if sys.platform == "darwin":
        try:
            from solhunter_zero import macos_setup

            macos_setup.ensure_network()
        except SystemExit:
            return False, "macOS network check failed"
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"Network error: {exc}"
        return True, "Network connectivity OK"
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


def run_basic_checks(min_bytes: int = 1 << 30, url: str | None = None) -> None:
    """Run minimal startup checks for disk space and internet connectivity.

    Parameters
    ----------
    min_bytes:
        Minimum free disk space required for the application to run.
    url:
        Optional URL to test for network reachability. Defaults to the
        ``SOLANA_RPC_URL`` environment variable or the public Solana RPC
        endpoint when unset.
    """

    check_disk_space(min_bytes)
    check_internet(url)

