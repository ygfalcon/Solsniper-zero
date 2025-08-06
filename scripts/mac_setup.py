#!/usr/bin/env python3
"""Install macOS dependencies for SolHunter Zero."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess[str]:
    """Run command printing it."""
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True, **kwargs)


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


def _apply_brew_env() -> None:
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


def ensure_homebrew() -> None:
    if shutil.which("brew") is None:
        print("Homebrew not found. Installing...")
        _run([
            "/bin/bash",
            "-c",
            "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)",
        ])
    _apply_brew_env()


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
    _apply_brew_env()


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
    content = profile.read_text() if profile.exists() else ""
    brew_env = subprocess.check_output(["brew", "shellenv"], text=True)
    if "HOMEBREW_PREFIX" not in content:
        with profile.open("a") as fh:
            fh.write(brew_env)
            fh.write("\n")
        print(f"Updated {profile} with Homebrew environment.")
    cargo_line = 'source "$HOME/.cargo/env"'
    if cargo_line not in content:
        with profile.open("a") as fh:
            fh.write(cargo_line + "\n")


def upgrade_pip_and_torch() -> None:
    if shutil.which("python3.11") is None:
        return
    _run(["python3.11", "-m", "pip", "install", "--upgrade", "pip"], check=False)
    _run(
        [
            "python3.11",
            "-m",
            "pip",
            "install",
            "torch==2.1.0",
            "torchvision==0.16.0",
            "--extra-index-url",
            "https://download.pytorch.org/whl/metal",
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-interactive", action="store_true")
    args = parser.parse_args(argv)
    if platform.system() != "Darwin":
        raise SystemExit("mac_setup is only intended for macOS")
    ensure_xcode(args.non_interactive)
    ensure_homebrew()
    install_brew_packages()
    ensure_rustup()
    upgrade_pip_and_torch()
    verify_tools()
    ensure_profile()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
