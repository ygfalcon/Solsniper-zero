#!/usr/bin/env python3
"""Interactive helper to create or update config.toml for basic setup."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
import sys
import tomllib
import argparse

try:
    import tomli_w  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    print("The 'tomli-w' package is required. Install it with 'pip install tomli-w'.", file=sys.stderr)
    sys.exit(1)

CONFIG_PATH = Path("config.toml")
EXAMPLE_PATH = Path("config.example.toml")


def load_config() -> dict:
    if CONFIG_PATH.is_file():
        with CONFIG_PATH.open("rb") as fh:
            return tomllib.load(fh)
    if EXAMPLE_PATH.is_file():
        shutil.copy(EXAMPLE_PATH, CONFIG_PATH)
        with CONFIG_PATH.open("rb") as fh:
            return tomllib.load(fh)
    return {}


def save_config(cfg: dict) -> None:
    with CONFIG_PATH.open("wb") as fh:
        fh.write(tomli_w.dumps(cfg).encode("utf-8"))


PROMPTS = [
    ("birdeye_api_key", "BIRDEYE_API_KEY", "BirdEye API key"),
    ("solana_rpc_url", "SOLANA_RPC_URL", "Solana RPC URL"),
    ("dex_base_url", "DEX_BASE_URL", "Base DEX URL (mainnet)"),
    ("dex_testnet_url", "DEX_TESTNET_URL", "DEX testnet URL"),
    ("orca_dex_url", "ORCA_DEX_URL", "Orca DEX URL"),
    ("raydium_dex_url", "RAYDIUM_DEX_URL", "Raydium DEX URL"),
]


# Defaults applied when --auto is used. These point to public endpoints but
# can be overridden in the generated config for custom providers.
AUTO_DEFAULTS = {
    "dex_base_url": "https://quote-api.jup.ag",
    "dex_testnet_url": "https://quote-api.jup.ag",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive helper to create or update config.toml for basic setup."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Populate missing values with defaults without prompting.",
    )
    args = parser.parse_args()

    cfg = load_config()
    updated = False
    for key, env, desc in PROMPTS:
        val = os.getenv(env) or cfg.get(key)
        missing = val in (None, "", "YOUR_BIRDEYE_KEY")
        if missing:
            if args.auto and key in AUTO_DEFAULTS:
                cfg[key] = AUTO_DEFAULTS[key]
                updated = True
                continue
            try:
                inp = input(f"Enter {desc}: ").strip()
            except EOFError:
                inp = ""
            if inp:
                cfg[key] = inp
                updated = True
    if updated:
        save_config(cfg)
        print(f"Configuration saved to {CONFIG_PATH}")
    else:
        print("Config already contains required values.")


if __name__ == "__main__":
    main()
