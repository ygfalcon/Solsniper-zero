"""Utilities for analyzing on-chain activity."""

from __future__ import annotations

import logging
import os
from typing import List, Dict

import requests

try:
    from solana.publickey import PublicKey  # type: ignore
except Exception:  # pragma: no cover - minimal stub when solana is missing
    class PublicKey(str):
        def __new__(cls, value: str):
            return str.__new__(cls, value)

    import types, sys

    mod = types.ModuleType("solana.publickey")
    mod.PublicKey = PublicKey
    sys.modules.setdefault("solana.publickey", mod)

from solana.rpc.api import Client

from .scanner_onchain import scan_tokens_onchain
from .exchange import DEX_BASE_URL

logger = logging.getLogger(__name__)

# REST endpoints for real-time market data
LIQ_PATH = "/v1/liquidity"
DEPTH_PATH = "/v1/depth"
VOLUME_PATH = "/v1/volume"


def _tx_volume(entries: List[dict]) -> float:
    """Return the total volume represented by ``entries``.

    Each entry is expected to contain an ``amount`` field.  Entries missing the
    field contribute ``0.0`` to the total.
    """

    return float(sum(e.get("amount", 0.0) for e in entries))


def top_volume_tokens(rpc_url: str, limit: int = 10) -> list[str]:
    """Return the addresses of the highest volume tokens.

    Parameters
    ----------
    rpc_url:
        RPC endpoint for querying the blockchain.
    limit:
        Maximum number of token addresses to return.
    """

    if not rpc_url:
        raise ValueError("rpc_url is required")

    tokens = scan_tokens_onchain(rpc_url)
    if not tokens:
        return []

    client = Client(rpc_url)
    vol_list: list[tuple[str, float]] = []
    for tok in tokens:
        try:
            resp = client.get_signatures_for_address(PublicKey(tok))
            entries = resp.get("result", [])
            volume = _tx_volume(entries)
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch tx for %s: %s", tok, exc)
            volume = 0.0
        vol_list.append((tok, volume))

    vol_list.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _v in vol_list[:limit]]


def fetch_dex_metrics(token: str, base_url: str | None = None) -> Dict[str, float]:
    """Return real-time liquidity, orderbook depth and volume for ``token``.

    Parameters
    ----------
    token:
        Token mint address or symbol.
    base_url:
        Base URL of the DEX REST API.  Defaults to ``DEX_BASE_URL``.
    """

    base = base_url or DEX_BASE_URL
    metrics = {"liquidity": 0.0, "depth": 0.0, "volume": 0.0}
    if not token:
        return metrics

    for path, key in (
        (LIQ_PATH, "liquidity"),
        (DEPTH_PATH, "depth"),
        (VOLUME_PATH, "volume"),
    ):
        url = f"{base.rstrip('/')}{path}?token={token}"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            val = data.get(key)
            if isinstance(val, (int, float)):
                metrics[key] = float(val)
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch %s for %s: %s", key, token, exc)

    return metrics

