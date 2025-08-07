#!/usr/bin/env python3
"""Install macOS dependencies for SolHunter Zero."""

from __future__ import annotations

import argparse
import os
from solhunter_zero import platform_utils
import shutil
import subprocess
import sys
import time
from pathlib import Path
from collections.abc import Callable
from urllib import request

try:
    from solhunter_zero.device import (
        METAL_EXTRA_INDEX,
        TORCH_METAL_VERSION,
        TORCHVISION_METAL_VERSION,
    )
except Exception:  # pragma: no cover - optional import for CI
    METAL_EXTRA_INDEX = []
    TORCH_METAL_VERSION = ""
    TORCHVISION_METAL_VERSION = ""


ROOT = Path(__file__).resolve().parent.parent
MAC_SETUP_MARKER = ROOT / ".cache" / "mac_setup_complete"


def _run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess[str]:
    """Run command printing it."""
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True, **kwargs)


def ensure_network() -> None:
    """Abort if no network connectivity is available."""
    try:
        subprocess.run(
            ["ping", "-c", "1", "1.1.1.1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return
    except Exception:
        try:
            with request.urlopen("https://example.com", timeout=5):
                return
        except Exception:
            print(
                "Network check failed. Please connect to the internet and re-run this script.",
                file=sys.stderr,
            )
            raise SystemExit(1)


def ensure_xcode(non_interactive: bool) -> None:
    if subprocess.run(["xcode-select", "-p"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return
    print("Installing Xcode command line tools...")
    subprocess.run(["xcode-select", "--install"], check=False)
    elapsed = 0
    timeout = 300
    interval = 10
    while subprocess.run(["xcode-select", "-p"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        if non_interactive:
            if elapsed >= timeout:
                print(f"Command line tools installation timed out after {timeout}s.", file=sys.stderr)
                raise SystemExit(1)
            time.sleep(interval)
            elapsed += interval
        else:
            ans = input(
                "Command line tools not yet installed. Press Enter to re-check or type 'c' to cancel: "
            )
            if ans.lower() == "c":
                print("Please re-run this script after the tools are installed.")
                raise SystemExit(1)

def apply_brew_env() -> None:
    """Load Homebrew environment variables into ``os.environ``."""
    try:
        out = subprocess.check_output(["brew", "shellenv"], text=True)
    except Exception:
        return
    for line in out.splitlines():
        if not line.startswith("export "):
            continue
        key, val = line[len("export ") :].split("=", 1)
        val = val.strip().strip('"')
        if key == "PATH":
            os.environ[key] = f"{val}:{os.environ.get(key, '')}"
        else:
            os.environ[key] = val


# Backwards compatibility
_apply_brew_env = apply_brew_env


def ensure_homebrew() -> None:
    if shutil.which("brew") is None:
        print("Homebrew not found. Installing...")
        _run([
            "/bin/bash",
            "-c",
            "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)",
        ])
    apply_brew_env()


def install_brew_packages() -> None:
    _run(["brew", "update"], check=False)
    _run(
        [
            "brew",
            "install",
            "python@3.11",
            "rustup-init",
            "pkg-config",
            "cmake",
            "protobuf",
        ],
        check=False,
    )
    apply_brew_env()


def ensure_rustup() -> None:
    if shutil.which("rustup") is None:
        _run(["rustup-init", "-y"])
        cargo_env = Path.home() / ".cargo" / "env"
        if cargo_env.exists():
            with cargo_env.open() as fh:
                for line in fh:
                    if line.startswith("export PATH="):
                        value = line.split("=", 1)[1].strip().strip('"')
                        os.environ["PATH"] = f"{value}:{os.environ.get('PATH', '')}"
                        break


def ensure_profile() -> None:
    shell = os.environ.get("SHELL", "")
    profile = Path.home() / (".zprofile" if shell.endswith("zsh") else ".bash_profile")
    content = profile.read_text().splitlines() if profile.exists() else []
    brew_env = subprocess.check_output(["brew", "shellenv"], text=True).splitlines()

    for line in brew_env:
        if line not in content:
            with profile.open("a") as fh:
                fh.write(line + "\n")
            print(f"Success: added line to {profile}: {line}")
            content.append(line)
        else:
            print(f"No change needed for {profile}: {line}")

    cargo_line = 'source "$HOME/.cargo/env"'
    if cargo_line not in content:
        with profile.open("a") as fh:
            fh.write(cargo_line + "\n")
        print(f"Success: added line to {profile}: {cargo_line}")
    else:
        print(f"No change needed for {profile}: {cargo_line}")


def upgrade_pip_and_torch() -> None:
    if shutil.which("python3.11") is None:
        return
    _run(["python3.11", "-m", "pip", "install", "--upgrade", "pip"], check=False)
    if not TORCH_METAL_VERSION or not TORCHVISION_METAL_VERSION:
        return
    _run(
        [
            "python3.11",
            "-m",
            "pip",
            "install",
            f"torch=={TORCH_METAL_VERSION}",
            f"torchvision=={TORCHVISION_METAL_VERSION}",
            *METAL_EXTRA_INDEX,
        ],
        check=False,
    )


def verify_tools() -> None:
    missing = []
    for tool in ["python3.11", "brew", "rustup"]:
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        brew_prefix = subprocess.check_output(["brew", "--prefix"], text=True).strip()
        print(
            f"Missing {' '.join(missing)} on PATH. Ensure {brew_prefix}/bin is in your PATH and re-run this script."
        )
        raise SystemExit(1)


MANUAL_FIXES = {
    "xcode": "Install Xcode command line tools with 'xcode-select --install' and rerun the script.",
    "homebrew": "Install Homebrew from https://brew.sh and ensure it is on your PATH.",
    "brew_packages": "Run 'brew install python@3.11 rustup-init pkg-config cmake protobuf'.",
    "rustup": "Run 'rustup-init -y' and ensure '$HOME/.cargo/bin' is on your PATH.",
    "pip_torch": (
        "Ensure python3.11 is installed then run 'python3.11 -m pip install --upgrade pip '"
        f"'torch=={TORCH_METAL_VERSION} torchvision=={TORCHVISION_METAL_VERSION} {' '.join(METAL_EXTRA_INDEX)}'."
    ),
    "verify_tools": "Ensure Homebrew's bin directory is on PATH and re-run this script.",
    "profile": (
        "Add 'eval $(brew shellenv)' and 'source \"$HOME/.cargo/env\"' to your shell profile"
        " so the tools are available in new shells."
    ),
}


def prepare_macos_env(
    non_interactive: bool = True, force: bool = False
) -> dict[str, object]:
    """Ensure core macOS development tools are installed.

    Returns a report dictionary describing the status of each setup step.
    The returned mapping contains a ``steps`` dict with per-step ``status`` and
    an overall ``success`` boolean. If a cache marker exists and ``force`` is
    ``False`` the installation steps are skipped.
    """

    ensure_network()

    report: dict[str, object] = {"steps": {}, "success": True}

    if MAC_SETUP_MARKER.exists() and not force:
        return report

    if force and MAC_SETUP_MARKER.exists():
        MAC_SETUP_MARKER.unlink(missing_ok=True)

    steps: list[tuple[str, Callable[[], None]]] = [
        ("xcode", lambda: ensure_xcode(non_interactive)),
        ("homebrew", ensure_homebrew),
        ("brew_packages", install_brew_packages),
        ("rustup", ensure_rustup),
        ("pip_torch", upgrade_pip_and_torch),
        ("verify_tools", verify_tools),
        ("profile", ensure_profile),
    ]

    for idx, (name, func) in enumerate(steps):
        try:
            func()
        except SystemExit as exc:
            if exc.code:
                report["steps"][name] = {
                    "status": "error",
                    "message": str(exc),
                }
                report["success"] = False
                # mark remaining steps as skipped
                for n, _ in steps[idx + 1 :]:
                    report["steps"][n] = {"status": "skipped"}
                break
        except Exception as exc:  # pragma: no cover - unexpected failure
            report["steps"][name] = {"status": "error", "message": str(exc)}
            report["success"] = False
            for n, _ in steps[idx + 1 :]:
                report["steps"][n] = {"status": "skipped"}
            break
        else:
            report["steps"][name] = {"status": "ok"}

    if report.get("success"):
        MAC_SETUP_MARKER.parent.mkdir(parents=True, exist_ok=True)
        MAC_SETUP_MARKER.write_text("ok")

    return report


def ensure_tools(*, non_interactive: bool = True) -> dict[str, object]:
    """Ensure essential macOS development tools are present.

    On Apple Silicon Macs this checks for Homebrew, Xcode command line tools,
    Python 3.11 and rustup.  If any are missing ``prepare_macos_env`` is
    invoked to install them.  A report dictionary is returned mirroring
    ``prepare_macos_env``'s output.
    """

    if not platform_utils.is_macos_arm64():
        return {"steps": {}, "success": True}

    missing_tools: list[str] = []
    try:
        if (
            subprocess.run(
                ["xcode-select", "-p"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            != 0
        ):
            missing_tools.append("xcode-select")
    except FileNotFoundError:
        missing_tools.append("xcode-select")
    for cmd in ("brew", "python3.11", "rustup"):
        if shutil.which(cmd) is None:
            missing_tools.append(cmd)
    if not missing_tools:
        return {"steps": {}, "success": True}

    print(
        "Missing macOS tools: " + ", ".join(missing_tools) + ". Running mac setup..."
    )
    report = prepare_macos_env(non_interactive=non_interactive, force=True)
    apply_brew_env()
    for step, info in report["steps"].items():
        msg = info.get("message", "")
        if msg:
            print(f"{step}: {info['status']} - {msg}")
        else:
            print(f"{step}: {info['status']}")
    if not report.get("success"):
        print(
            "macOS environment preparation failed; continuing without required tools",
            file=sys.stderr,
        )
        for step, info in report["steps"].items():
            if info.get("status") == "error":
                fix = MANUAL_FIXES.get(step)
                if fix:
                    print(f"Manual fix for {step}: {fix}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    report = prepare_macos_env(args.non_interactive, force=args.force)
    for step, info in report["steps"].items():
        msg = info.get("message", "")
        if msg:
            print(f"{step}: {info['status']} - {msg}")
        else:
            print(f"{step}: {info['status']}")
    if not report.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
