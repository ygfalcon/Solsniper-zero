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


def ensure_keypair(auto: bool = False) -> None:
    """Ensure an active keypair exists and has SOL available.

    When a keypair is present but unfunded the user is prompted to fund it
    unless ``auto`` is ``True`` in which case the script exits with
    instructions.
    """

    from solhunter_zero import wallet

    def _balance(pubkey: str) -> float:
        """Return the SOL balance for ``pubkey`` using the ``solana`` CLI.

        Falls back to a simple JSON-RPC request if the CLI is unavailable.
        """

        try:  # Prefer the ``solana`` CLI if present
            out = subprocess.check_output(["solana", "balance", pubkey], text=True)
            return float(out.split()[0])
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            import json
            import urllib.request

            url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
            payload = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBalance",
                "params": [pubkey],
            }).encode()
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as resp:  # noqa: S310 - trusted endpoint
                data = json.load(resp)
            lamports = data.get("result", {}).get("value", 0)
            return lamports / 1_000_000_000

    if not wallet.list_keypairs():
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

    name = wallet.get_active_keypair_name()
    if not name:
        available = wallet.list_keypairs()
        if not available:
            raise SystemExit("No keypair available; cannot continue.")
        wallet.select_keypair(available[0])
        name = available[0]

    keypair = wallet.load_selected_keypair()
    if keypair is None:
        raise SystemExit("Failed to load active keypair.")

    pubkey = str(keypair.pubkey())
    bal = _balance(pubkey)
    if bal <= 0:
        msg = (
            f"Active keypair '{name}' ({pubkey}) has zero SOL. "
            "Fund this account and rerun the script."
        )
        if auto:
            print(msg)
            raise SystemExit(1)
        print(msg)
        input("Press Enter once funds have been deposited...")
        if _balance(pubkey) <= 0:
            raise SystemExit("No funds detected; aborting.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Guided setup and launch")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency check")
    parser.add_argument("--skip-setup", action="store_true", help="Skip config and wallet prompts")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Exit with instructions instead of prompting when funds are missing",
    )
    args, rest = parser.parse_known_args(argv)

    if not args.skip_deps:
        ensure_deps()
    if not args.skip_setup:
        ensure_config()
        ensure_keypair(auto=args.auto)

    os.execv("./run.sh", ["./run.sh", "--auto", *rest])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
