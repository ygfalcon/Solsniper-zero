#!/usr/bin/env python3
"""Environment preflight checks for Solsniper-zero."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Tuple
from urllib import error, request

from scripts.env_checks import (
    Check,
    check_dependencies,
    check_homebrew,
    check_python_version,
    check_rust_toolchain,
    check_rustup,
)


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


def check_network(default_url: str = "https://api.mainnet-beta.solana.com") -> Check:
    url = os.environ.get("SOLANA_RPC_URL", default_url)
    try:
        req = request.Request(url, method="HEAD")
        with request.urlopen(req, timeout=5):
            pass
    except error.URLError as exc:
        return False, f"Network error: {exc}"
    return True, f"Network access to {url} OK"


def check_gpu() -> Check:
    try:
        from solhunter_zero import device
        try:
            import torch  # type: ignore[import]
        except Exception:  # pragma: no cover - torch is optional
            torch = None  # type: ignore
        if (
            platform.system() == "Darwin"
            and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1"
        ):
            return False, "PYTORCH_ENABLE_MPS_FALLBACK=1 not set"
        if not device.detect_gpu():
            return False, "No GPU backend detected"
        if torch is not None and torch.backends.mps.is_available():
            try:
                import torch
                torch.set_default_device("mps")
            except Exception as exc:
                return False, (
                    f"Metal available but failed to set default device: {exc}"
                )
            return True, "Metal GPU available"
        return True, "CUDA GPU available"
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


def main() -> None:
    failures: List[Tuple[str, str]] = []
    for name, func in CHECKS:
        ok, msg = func()
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
