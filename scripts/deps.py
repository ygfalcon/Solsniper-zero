#!/usr/bin/env python3
"""Dependency checking utilities for SolHunter Zero."""

from __future__ import annotations

import argparse
import json
import pkgutil
import platform
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - should not happen
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parent.parent

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
    missing_required: list[str] = []
    for dep in deps:
        mod = re.split("[<=>]", dep)[0].replace("-", "_")
        if pkgutil.find_loader(mod) is None:
            missing_required.append(mod)
    missing_optional = [m for m in OPTIONAL_DEPS if pkgutil.find_loader(m) is None]
    return missing_required, missing_optional


def ensure_deps() -> None:
    """Install any missing required or optional dependencies."""

    req, opt = check_deps()
    if not req and not opt:
        return

    if req:
        print("Installing required dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", ".[uvloop]"])
        except subprocess.CalledProcessError as exc:  # pragma: no cover - hard failure
            print(f"Failed to install required dependencies: {exc}")
            raise SystemExit(exc.returncode)

    if "torch" in opt and platform.system() == "Darwin" and platform.machine() == "arm64":
        print(
            "Installing torch==2.1.0 and torchvision==0.16.0 for macOS arm64 with Metal support..."
        )
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    # Pinned versions: update together when upgrading Metal wheels.
                    "torch==2.1.0",
                    "torchvision==0.16.0",
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/metal",
                ]
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - network failure
            print(f"Failed to install torch with Metal wheels: {exc}")
            raise SystemExit(exc.returncode)
        else:
            opt.remove("torch")

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        import torch

        if not torch.backends.mps.is_available():
            print("MPS backend not available; ensure Metal wheel is installed")
            raise SystemExit(1)

    if opt:
        print("Installing optional dependencies...")
        mapping = {
            "faiss": "faiss-cpu",
            "sentence_transformers": "sentence-transformers",
            "torch": "torch",
            "orjson": "orjson",
            "lz4": "lz4",
            "zstandard": "zstandard",
            "msgpack": "msgpack",
        }
        mods = set(opt)
        extras: list[str] = []
        if "orjson" in mods:
            extras.append("fastjson")
        if {"lz4", "zstandard"} & mods:
            extras.append("fastcompress")
        if "msgpack" in mods:
            extras.append("msgpack")
        if extras:
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        f".[{','.join(extras)}]",
                    ]
                )
            except subprocess.CalledProcessError as exc:  # pragma: no cover - network failure
                print(f"Failed to install optional extras '{extras}': {exc}")
                raise SystemExit(exc.returncode)
        remaining = mods - {"orjson", "lz4", "zstandard", "msgpack"}
        for name in remaining:
            pkg = mapping.get(name, name.replace("_", "-"))
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except subprocess.CalledProcessError as exc:  # pragma: no cover - network failure
                print(f"Failed to install optional dependency '{pkg}': {exc}")
                raise SystemExit(exc.returncode)

    req_after, opt_after = check_deps()
    if req_after or opt_after:
        if req_after:
            print(
                "Missing required dependencies after installation: "
                + ", ".join(req_after)
            )
        if opt_after:
            print(
                "Missing optional dependencies after installation: "
                + ", ".join(opt_after)
            )
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install", action="store_true", help="Install missing dependencies"
    )
    args = parser.parse_args(argv)

    if args.install:
        ensure_deps()
        return 0

    req, opt = check_deps()
    print(json.dumps({"required": req, "optional": opt}))
    return 1 if req or opt else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
