#!/usr/bin/env python3
"""Interactive startup script for SolHunter Zero."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

os.environ.setdefault("DEPTH_SERVICE", "true")


def check_deps() -> tuple[list[str], list[str]]:
    import pkgutil
    import re
    import tomllib

    with open("pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    deps = data.get("project", {}).get("dependencies", [])
    missing_required: list[str] = []
    for dep in deps:
        mod = re.split("[<=>]", dep)[0].replace("-", "_")
        if pkgutil.find_loader(mod) is None:
            missing_required.append(mod)
    optional = ["faiss", "sentence_transformers", "torch"]
    missing_optional = [m for m in optional if pkgutil.find_loader(m) is None]
    return missing_required, missing_optional


def ensure_deps() -> None:
    req, opt = check_deps()
    if req or opt:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", ".[uvloop]"])


def ensure_config() -> None:
    if not any(Path(name).is_file() for name in ("config.toml", "config.yaml", "config.yml")):
        from scripts import quick_setup

        quick_setup.main()


def ensure_keypair() -> None:
    from solhunter_zero import wallet

    if wallet.list_keypairs():
        return

    print("No keypairs found in 'keypairs/' directory.")
    path = input("Path to keypair JSON (leave blank for mnemonic): ").strip()
    if path:
        subprocess.check_call(["solhunter-wallet", "save", "default", path])
    else:
        mnemonic = input("Enter mnemonic: ").strip()
        if not mnemonic:
            print("No mnemonic provided. Run scripts/setup_default_keypair.sh for automated setup.")
            return
        passphrase = input("Passphrase (leave blank if none): ").strip()
        subprocess.check_call([
            "solhunter-wallet",
            "derive",
            "default",
            mnemonic,
            "--passphrase",
            passphrase,
        ])
    subprocess.check_call(["solhunter-wallet", "select", "default"])
    print("Keypair saved. You can use scripts/setup_default_keypair.sh for automated setup.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Guided setup and launch")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency check")
    parser.add_argument("--skip-setup", action="store_true", help="Skip config and wallet prompts")
    args, rest = parser.parse_known_args(argv)

    if not args.skip_deps:
        ensure_deps()
    if not args.skip_setup:
        ensure_config()
        ensure_keypair()

    os.execv("./run.sh", ["./run.sh", "--auto", *rest])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
