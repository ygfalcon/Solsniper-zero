from __future__ import annotations

import os
import json
import time
import urllib.request

import solhunter_zero.bootstrap_utils as bootstrap_utils
from solhunter_zero.config import apply_env_overrides, load_config
from solhunter_zero.paths import ROOT

__all__ = ["disk_space_required_bytes", "ensure_rpc", "ensure_cargo"]


def disk_space_required_bytes() -> int:
    """Return the minimum free bytes required based on configuration."""
    try:
        cfg = apply_env_overrides(load_config())
        limit_gb = float(cfg.get("offline_data_limit_gb", 50))
    except Exception:
        limit_gb = 50
    return int(limit_gb * (1024 ** 3))


def ensure_rpc(*, warn_only: bool = False) -> None:
    """Send a simple JSON-RPC request to ensure the Solana RPC is reachable."""
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    if not os.environ.get("SOLANA_RPC_URL"):
        print(f"Using default RPC URL {rpc_url}")
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "getHealth"}).encode()
    req = urllib.request.Request(
        rpc_url, data=payload, headers={"Content-Type": "application/json"}
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
                resp.read()
                break
        except Exception as exc:  # pragma: no cover - network failure
            if attempt == 2:
                msg = (
                    f"Failed to contact Solana RPC at {rpc_url} after 3 attempts: {exc}."
                    " Please ensure the endpoint is reachable or set SOLANA_RPC_URL to a valid RPC."
                )
                if warn_only:
                    print(f"Warning: {msg}")
                    return
                print(msg)
                raise SystemExit(1)
            wait = 2 ** attempt
            print(
                f"Attempt {attempt + 1} failed to contact Solana RPC at {rpc_url}: {exc}.",
                f" Retrying in {wait} seconds...",
            )
            time.sleep(wait)


def ensure_cargo() -> None:
    """Wrapper around :func:`bootstrap_utils.ensure_cargo` that syncs ROOT."""
    bootstrap_utils.ROOT = ROOT
    bootstrap_utils.ensure_cargo()
