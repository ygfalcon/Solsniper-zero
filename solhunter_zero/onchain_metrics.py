"""Utilities for analyzing on-chain activity."""

from __future__ import annotations

import logging
import os
from typing import List, Dict

import aiohttp
import asyncio

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

from .scanner_onchain import (
    scan_tokens_onchain,
    fetch_mempool_tx_rate,
    fetch_whale_wallet_activity,
    fetch_average_swap_size,
)
from .exchange import DEX_BASE_URL
from .http import get_session

logger = logging.getLogger(__name__)

# REST endpoints for real-time market data
LIQ_PATH = "/v1/liquidity"
DEPTH_PATH = "/v1/depth"
VOLUME_PATH = "/v1/volume"

_DEPTH_CACHE: Dict[str, float] = {}


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


async def fetch_dex_metrics_async(token: str, base_url: str | None = None) -> Dict[str, float]:
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

    session = await get_session()
    for path, key in (
        (LIQ_PATH, "liquidity"),
        (DEPTH_PATH, "depth"),
        (VOLUME_PATH, "volume"),
    ):
        url = f"{base.rstrip('/')}{path}?token={token}"
        try:
            async with session.get(url, timeout=5) as resp:
                resp.raise_for_status()
                data = await resp.json()
            val = data.get(key)
            if isinstance(val, (int, float)):
                metrics[key] = float(val)
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch %s for %s: %s", key, token, exc)

    return metrics


def fetch_dex_metrics(token: str, base_url: str | None = None) -> Dict[str, float]:
    """Synchronous wrapper for :func:`fetch_dex_metrics_async`."""
    return asyncio.run(fetch_dex_metrics_async(token, base_url))


def fetch_liquidity_onchain(token: str, rpc_url: str) -> float:
    """Return token liquidity computed from on-chain data.

    The function queries ``get_token_largest_accounts`` and sums the
    returned balances.  Failures result in ``0.0`` being returned.
    """

    if not rpc_url:
        raise ValueError("rpc_url is required")

    client = Client(rpc_url)
    try:
        resp = client.get_token_largest_accounts(PublicKey(token))
        accounts = resp.get("result", {}).get("value", [])
        total = 0.0
        for acc in accounts:
            val = acc.get("uiAmount")
            if isinstance(val, (int, float)):
                total += float(val)
            else:
                val = acc.get("amount")
                if isinstance(val, (int, float, str)):
                    try:
                        total += float(val)
                    except Exception:
                        pass
        return total
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch liquidity for %s: %s", token, exc)
        return 0.0


def fetch_volume_onchain(token: str, rpc_url: str) -> float:
    """Return recent transaction volume for ``token`` using ``get_signatures_for_address``."""

    if not rpc_url:
        raise ValueError("rpc_url is required")

    client = Client(rpc_url)
    try:
        resp = client.get_signatures_for_address(PublicKey(token))
        entries = resp.get("result", [])
        return _tx_volume(entries)
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch volume for %s: %s", token, exc)
        return 0.0


def fetch_token_age(token: str, rpc_url: str, *, limit: int = 20) -> float:
    """Return approximate age of ``token`` in seconds."""

    if not rpc_url:
        raise ValueError("rpc_url is required")

    client = Client(rpc_url)
    try:
        resp = client.get_signatures_for_address(PublicKey(token), limit=limit)
        entries = resp.get("result", [])
        times = [e.get("blockTime") for e in entries if e.get("blockTime")]
        if times:
            first = min(times)
            import time as _time

            return max(_time.time() - float(first), 0.0)
        return 0.0
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch token age for %s: %s", token, exc)
        return 0.0


def fetch_slippage_onchain(token: str, rpc_url: str) -> float:
    """Estimate slippage based on token account distribution."""

    if not rpc_url:
        raise ValueError("rpc_url is required")

    client = Client(rpc_url)
    try:
        resp = client.get_token_largest_accounts(PublicKey(token))
        accounts = resp.get("result", {}).get("value", [])
        if len(accounts) < 2:
            return 0.0
        first = accounts[0].get("uiAmount", accounts[0].get("amount", 0))
        second = accounts[1].get("uiAmount", accounts[1].get("amount", 0))
        try:
            first_val = float(first)
            second_val = float(second)
        except Exception:
            return 0.0
        return (first_val - second_val) / first_val if first_val else 0.0
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch slippage for %s: %s", token, exc)
        return 0.0


def order_book_depth_change(token: str, base_url: str | None = None) -> float:
    """Return the recent change in order-book depth for ``token``."""

    metrics = fetch_dex_metrics(token, base_url)
    depth = metrics.get("depth", 0.0)
    prev = _DEPTH_CACHE.get(token)
    _DEPTH_CACHE[token] = depth
    if prev is None:
        return 0.0
    return depth - prev


def collect_onchain_insights(
    token: str, rpc_url: str, base_url: str | None = None
) -> Dict[str, float]:
    """Return a dictionary with depth change, tx rate and whale activity."""

    return {
        "depth_change": order_book_depth_change(token, base_url),
        "tx_rate": fetch_mempool_tx_rate(token, rpc_url),
        "whale_activity": fetch_whale_wallet_activity(token, rpc_url),
        "avg_swap_size": fetch_average_swap_size(token, rpc_url),
    }

