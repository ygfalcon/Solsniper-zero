#!/usr/bin/env python3
"""Interactive helper to create or update config.toml for basic setup."""
from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Fill missing values from environment or defaults without prompting",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    example_cfg: dict[str, object] = {}
    if args.auto and EXAMPLE_PATH.is_file():
        with EXAMPLE_PATH.open("rb") as fh:
            example_cfg = tomllib.load(fh)

    updated = False
    for key, env, desc in PROMPTS:
        env_val = os.getenv(env)
        cfg_val = cfg.get(key)
        missing = cfg_val in (None, "", "YOUR_BIRDEYE_KEY")
        if args.auto:
            if missing:
                val = env_val or example_cfg.get(key, cfg_val or "")
                if cfg_val != val:
                    cfg[key] = val
                    updated = True
        else:
            if missing:
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
