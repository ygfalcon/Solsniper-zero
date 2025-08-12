#!/usr/bin/env python3
"""Environment setup utility for Solsniper-zero.

This script bootstraps a development environment by ensuring a
virtual environment is active, installing dependencies and tooling,
building the Rust FFI extension, and reporting next‑steps to the user.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import venv

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], **kwargs) -> None:
    """Run a subprocess and stream output."""
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def ensure_venv() -> None:
    """Ensure we're running inside a virtualenv, creating one if needed."""
    if sys.prefix == sys.base_prefix:
        venv_dir = ROOT / ".venv"
        if not venv_dir.exists():
            print(f"Creating venv at {venv_dir}")
            venv.create(venv_dir, with_pip=True)
        python_bin = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
        print(f"Re-executing under {python_bin}")
        os.execv(str(python_bin), [str(python_bin), __file__] + sys.argv[1:])


def pip_install(args: list[str]) -> None:
    run([sys.executable, "-m", "pip", "install", *args])


def install_project() -> None:
    extras = "fastjson,fastcompress,msgpack"
    packages = [
        f".[{extras}]",
        "protobuf",
        "grpcio-tools",
        "scikit-learn",
        "PyYAML",
    ]
    pip_install(["-e", *packages])


def install_brew_packages() -> None:
    if sys.platform != "darwin":
        return
    py_ver = f"python@{sys.version_info.major}.{sys.version_info.minor}"
    pkgs = [py_ver, "rustup", "pkg-config", "cmake", "protobuf"]
    run(["brew", "install", *pkgs])


def configure_pytorch(cfg: dict) -> None:
    torch_cfg = cfg.get("torch", {}) if isinstance(cfg, dict) else {}
    torch_ver = torch_cfg.get("torch_metal_version")
    tv_ver = torch_cfg.get("torchvision_metal_version")
    specs = []
    if torch_ver:
        specs.append(f"torch=={torch_ver}")
    else:
        specs.append("torch")
    if tv_ver:
        specs.append(f"torchvision=={tv_ver}")
    else:
        specs.append("torchvision")
    args = []
    if sys.platform == "darwin":
        args += ["--index-url", "https://download.pytorch.org/whl/metal"]
    args += specs
    pip_install(args)


def build_route_ffi() -> None:
    crate_dir = ROOT / "route_ffi"
    if not crate_dir.exists():
        return
    run(["cargo", "build", "--release", "--features=parallel"], cwd=crate_dir)
    target = crate_dir / "target" / "release"
    if sys.platform == "darwin":
        libname = "libroute_ffi.dylib"
    elif os.name == "nt":
        libname = "route_ffi.dll"
    else:
        libname = "libroute_ffi.so"
    src = target / libname
    dest = ROOT / "solhunter_zero" / libname
    if src.exists():
        shutil.copy2(src, dest)
        print(f"Copied {src} -> {dest}")
    else:
        print("Route FFI build artifact not found; set ROUTE_FFI_LIB manually.")


def preflight(cfg: dict) -> None:
    from solhunter_zero.preflight_utils import check_internet

    ok, msg = check_internet()
    if not ok:
        raise SystemExit(f"Internet check failed: {msg}")

    broker_url = os.environ.get("BROKER_URL") or cfg.get("broker_url") or cfg.get("broker_urls")
    if broker_url and "redis" in broker_url:
        if not shutil.which("redis-server"):
            raise SystemExit("Redis broker configured but redis-server not found.")


def print_summary() -> None:
    pkgs = [
        "protobuf",
        "grpcio-tools",
        "scikit-learn",
        "PyYAML",
        "torch",
        "torchvision",
    ]
    print("\nInstalled package versions:")
    for pkg in pkgs:
        try:
            ver = version(pkg)
        except PackageNotFoundError:
            ver = "not installed"
        print(f"  {pkg}: {ver}")
    print("\nNext steps:")
    print("  - export BROKER_WS_URLS")
    print("  - export BIRDEYE_API_KEY")


def load_config() -> dict:
    cfg_path = ROOT / "config.toml"
    if cfg_path.exists():
        with cfg_path.open("rb") as f:
            return tomllib.load(f)
    return {}


def main() -> None:
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")

    ensure_venv()
    cfg = load_config()
    install_brew_packages()
    preflight(cfg)
    install_project()
    configure_pytorch(cfg)
    build_route_ffi()
    print_summary()


if __name__ == "__main__":
    main()
