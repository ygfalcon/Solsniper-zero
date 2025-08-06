"""Utilities for building Rust components used by SolHunter Zero."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ensure_cargo() -> None:
    """Ensure the Rust toolchain and required build tools are available."""
    installed = False
    if shutil.which("cargo") is None:
        if shutil.which("curl") is None:
            print(
                "curl is required to install the Rust toolchain. "
                "Install it (e.g., with Homebrew: 'brew install curl') and re-run this script.",
            )
            raise SystemExit(1)
        if platform.system() == "Darwin":
            if shutil.which("brew") is None:
                print(
                    "Homebrew is required to install the Rust toolchain. "
                    "Install it by running scripts/mac_setup.sh and re-run this script.",
                )
                raise SystemExit(1)
            try:
                subprocess.check_call(
                    ["xcode-select", "-p"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                print("Xcode command line tools are required. Launching installer...")
                try:
                    subprocess.check_call(["xcode-select", "--install"])
                except subprocess.CalledProcessError as exc:  # pragma: no cover - hard failure
                    print(f"Failed to start Xcode command line tools installer: {exc}")
                else:
                    print(
                        "The installer may prompt for confirmation; "
                        "after it finishes, re-run this script to resume.",
                    )
                raise SystemExit(1)
        print("Installing Rust toolchain via rustup...")
        subprocess.check_call(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            shell=True,
        )
        installed = True
    cargo_bin = Path.home() / ".cargo" / "bin"
    os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    try:
        subprocess.check_call(["cargo", "--version"], stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        print("Failed to run 'cargo --version'. Is Rust installed correctly?")
        raise SystemExit(exc.returncode)
    if installed and platform.system() == "Darwin" and platform.machine() == "arm64":
        subprocess.check_call(["rustup", "target", "add", "aarch64-apple-darwin"])
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            targets = subprocess.check_output(["rustup", "target", "list"], text=True)
        except subprocess.CalledProcessError as exc:
            print("Failed to list rust targets. Is rustup installed correctly?")
            raise SystemExit(exc.returncode)
        if "aarch64-apple-darwin" not in targets:
            subprocess.check_call(["rustup", "target", "add", "aarch64-apple-darwin"])

    missing = [tool for tool in ("pkg-config", "cmake") if shutil.which(tool) is None]
    if missing:
        if platform.system() == "Darwin" and shutil.which("brew") is not None:
            print(f"Missing {', '.join(missing)}. Attempting to install with Homebrew...")
            try:
                subprocess.check_call(["brew", "install", "pkg-config", "cmake"])
            except subprocess.CalledProcessError as exc:
                print(f"Homebrew installation failed: {exc}")
            else:
                missing = [tool for tool in ("pkg-config", "cmake") if shutil.which(tool) is None]
        if missing:
            names = " and ".join(missing)
            brew = " ".join(missing)
            print(
                f"{names} {'are' if len(missing) > 1 else 'is'} required to build native extensions. "
                f"Install {'them' if len(missing) > 1 else 'it'} (e.g., with Homebrew: 'brew install {brew}') and re-run this script.",
            )
            raise SystemExit(1)


def build_route_ffi() -> None:
    """Build the ``route_ffi`` Rust library if missing."""
    libname = "libroute_ffi.dylib" if platform.system() == "Darwin" else "libroute_ffi.so"
    libpath = ROOT / "solhunter_zero" / libname
    if libpath.exists():
        return

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            installed_targets = subprocess.check_output(
                ["rustup", "target", "list", "--installed"], text=True
            )
        except subprocess.CalledProcessError as exc:
            print("Failed to verify rust targets. Is rustup installed correctly?")
            raise SystemExit(exc.returncode)
        if "aarch64-apple-darwin" not in installed_targets:
            subprocess.check_call(["rustup", "target", "add", "aarch64-apple-darwin"])

    cmd = [
        "cargo",
        "build",
        "--manifest-path",
        str(ROOT / "route_ffi" / "Cargo.toml"),
        "--release",
    ]
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        cmd.extend(["--target", "aarch64-apple-darwin"])
    subprocess.check_call(cmd)

    target_dir = ROOT / "route_ffi" / "target"
    candidates = [target_dir / "release" / libname]
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        candidates.append(target_dir / "aarch64-apple-darwin" / "release" / libname)

    for built in candidates:
        if built.exists():
            shutil.copy2(built, libpath)
            break
    if not libpath.exists():
        print(f"Warning: failed to locate built {libname}; please build manually.")
    elif platform.system() == "Darwin":
        try:
            subprocess.check_call([
                "codesign",
                "--force",
                "--sign",
                "-",
                str(libpath),
            ])
        except subprocess.CalledProcessError:
            print(
                "Warning: failed to codesign libroute_ffi.dylib; please run 'codesign --force --sign - solhunter_zero/libroute_ffi.dylib' manually."
            )


def build_depth_service() -> Path:
    """Build the ``depth_service`` binary if missing and return its path."""
    bin_path = ROOT / "target" / "release" / "depth_service"
    if bin_path.exists() and os.access(bin_path, os.X_OK):
        return bin_path
    try:
        subprocess.check_call(
            [
                "cargo",
                "build",
                "--manifest-path",
                str(ROOT / "depth_service" / "Cargo.toml"),
                "--release",
            ]
        )
    except FileNotFoundError:
        print(
            "cargo is not installed. Please run 'cargo build --manifest-path depth_service/Cargo.toml --release'",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not bin_path.exists() or not os.access(bin_path, os.X_OK):
        print(
            "depth_service binary missing or not executable after build. Please run 'cargo build --manifest-path depth_service/Cargo.toml --release'.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return bin_path
