from __future__ import annotations

import json
import os
import time
import urllib.request


def _check_internet(url: str) -> None:
    """Ensure basic internet connectivity by reaching a known host."""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # nosec B310
                resp.read()
                return
        except Exception as exc:  # pragma: no cover - network failure
            if attempt == 2:
                print(
                    f"Failed to reach {url} after 3 attempts: {exc}. "
                    "Check your internet connection."
                )
                raise SystemExit(1)
            wait = 2**attempt
            print(
                f"Attempt {attempt + 1} failed to reach {url}: {exc}. "
                f"Retrying in {wait} seconds..."
            )
            time.sleep(wait)


def _ensure_rpc(rpc_url: str, warn_only: bool) -> None:
    """Send a simple JSON-RPC request to ensure the Solana RPC is reachable."""
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
                return
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
            wait = 2**attempt
            print(
                f"Attempt {attempt + 1} failed to contact Solana RPC at {rpc_url}: {exc}.",
                f" Retrying in {wait} seconds...",
            )
            time.sleep(wait)


def verify_connectivity(
    url: str = "https://example.com",
    *,
    rpc_url: str | None = None,
    warn_only: bool = False,
) -> None:
    """Verify both internet and Solana RPC connectivity."""
    _check_internet(url)
    rpc_url = rpc_url or os.environ.get(
        "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
    )
    _ensure_rpc(rpc_url, warn_only)
