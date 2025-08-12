#!/usr/bin/env python3
"""Dependency checking utilities for SolHunter Zero."""

from __future__ import annotations

import argparse
import os
import platform
import pkgutil
import re
import shutil
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - should not happen
    import tomli as tomllib  # type: ignore

from solhunter_zero.paths import ROOT

# Map distribution names to the module names they provide when imported. This is
# necessary for packages whose import name differs from the name used on PyPI.
IMPORT_NAME_MAP = {
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "pytorch-lightning": "pytorch_lightning",
    "faiss-cpu": "faiss",
    "opencv-python": "cv2",
    "beautifulsoup4": "bs4",
    "grpcio-tools": "grpc_tools",
}

OPTIONAL_DEPS = [
    "faiss",
    "sentence_transformers",
    "torch",
    "orjson",
    "lz4",
    "zstandard",
    "msgpack",
]


def check_deps() -> tuple[list[str], list[str]]:
    """Return lists of missing required and optional modules."""
    with open(ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    deps = data.get("project", {}).get("dependencies", [])
    build_deps = data.get("build-system", {}).get("requires", [])
    for dep in build_deps:
        if dep.startswith("grpcio-tools"):
            deps.append(dep)
    missing_required: list[str] = []
    for dep in deps:
        dist_name = re.split("[<=>]", dep)[0]
        mod = IMPORT_NAME_MAP.get(dist_name, dist_name).replace("-", "_")
        if pkgutil.find_loader(mod) is None:
            missing_required.append(mod)

    missing_optional = []
    for dep in OPTIONAL_DEPS:
        mod = IMPORT_NAME_MAP.get(dep, dep).replace("-", "_")
        if pkgutil.find_loader(mod) is None:
            missing_optional.append(mod)

    return missing_required, missing_optional


def ensure_route_ffi_lib() -> Path | None:
    """Ensure the route_ffi dynamic library is discoverable.

    The build process places the compiled library in
    ``route_ffi/target/release``.  This helper copies it into the
    ``solhunter_zero`` package directory so the Python bindings can load it.
    When copying fails, ``ROUTE_FFI_LIB`` is set to point at the build
    artifact instead.  The resolved path is printed so users can verify the
    location.
    """

    libname = "libroute_ffi.dylib" if platform.system() == "Darwin" else "libroute_ffi.so"
    dest = ROOT / "solhunter_zero" / libname
    if dest.exists():
        os.environ.setdefault("ROUTE_FFI_LIB", str(dest))
        print(f"route_ffi library: {dest}")
        return dest

    src = ROOT / "route_ffi" / "target" / "release" / libname
    if src.exists():
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            os.environ.setdefault("ROUTE_FFI_LIB", str(dest))
            print(f"route_ffi library: {dest}")
            return dest
        except OSError as exc:
            os.environ["ROUTE_FFI_LIB"] = str(src)
            print(f"route_ffi library: {src} (copy failed: {exc})")
            return src

    env_path = os.environ.get("ROUTE_FFI_LIB")
    if env_path:
        print(f"route_ffi library: {env_path}")
        return Path(env_path)

    print("route_ffi library not found")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install project dependencies")
    parser.add_argument(
        "--install-optional",
        action="store_true",
        help="Install optional dependencies",
    )
    parser.add_argument(
        "--extras",
        nargs="*",
        help="Extras to install from the local package",
    )
    args = parser.parse_args(argv)

    from solhunter_zero.bootstrap_utils import DepsConfig, ensure_deps

    cfg = DepsConfig(
        install_optional=args.install_optional,
        extras=args.extras if args.extras else ("uvloop",),
    )
    ensure_deps(cfg)
    ensure_route_ffi_lib()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
