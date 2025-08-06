#!/usr/bin/env python3
"""Interactive helper to create or update config.toml for basic setup."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
import sys
import tomllib

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


def main() -> None:
    cfg = load_config()
    updated = False
    for key, env, desc in PROMPTS:
        val = os.getenv(env) or cfg.get(key)
        missing = val in (None, "", "YOUR_BIRDEYE_KEY", "YOUR_BIRDEYE_API_KEY")
        if missing:
            try:
                inp = input(f"Enter {desc}: ").strip()
            except EOFError:
                inp = ""
            if inp:
                cfg[key] = inp
                updated = True
            else:
                # remove empty values so they are not exported
                if key in cfg:
                    cfg.pop(key, None)
                    updated = True
    if updated:
        save_config(cfg)
        print(f"Configuration saved to {CONFIG_PATH}")
    else:
        print("Config already contains required values.")


if __name__ == "__main__":
    main()
