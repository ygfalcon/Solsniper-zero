from __future__ import annotations

import logging
import asyncio
import os
from typing import List, Dict, Any

from solders.pubkey import Pubkey
from solhunter_zero.lru import TTLCache


try:
    from solana.publickey import PublicKey  # type: ignore
except Exception:  # pragma: no cover - fallback when solana lacks PublicKey
    class PublicKey(str):
        """Minimal stand-in for ``solana.publickey.PublicKey``."""

        def __new__(cls, value: str):
            return str.__new__(cls, value)

    import types, sys
    mod = types.ModuleType("solana.publickey")
    mod.PublicKey = PublicKey
    sys.modules.setdefault("solana.publickey", mod)
from solana.rpc.api import Client
from solana.rpc.async_api import AsyncClient

logger = logging.getLogger(__name__)

TOKEN_PROGRAM_ID = PublicKey("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

# module level caches for on-chain metrics
METRIC_CACHE_TTL = 30  # seconds
MEMPOOL_RATE_CACHE = TTLCache(maxsize=256, ttl=METRIC_CACHE_TTL)
WHALE_ACTIVITY_CACHE = TTLCache(maxsize=256, ttl=METRIC_CACHE_TTL)
AVG_SWAP_SIZE_CACHE = TTLCache(maxsize=256, ttl=METRIC_CACHE_TTL)

# history of on-chain features used for mempool rate forecasting
MEMPOOL_FEATURE_HISTORY: Dict[tuple[str, str], list[list[float]]] = {}




async def scan_tokens_onchain(rpc_url: str, *, return_metrics: bool = False) -> List[str] | List[Dict[str, Any]]:
    """Query recent token accounts from the blockchain and return mints whose
    names end with ``bonk``.

    Parameters
    ----------
    rpc_url:
        Solana RPC endpoint.
    """
    if not rpc_url:
        raise ValueError("rpc_url is required")

    from . import onchain_metrics

    backoff = 1
    max_backoff = 60
    attempts = 0
    async with AsyncClient(rpc_url) as client:
        while True:
            try:
                resp = await client.get_program_accounts(
                    TOKEN_PROGRAM_ID, encoding="jsonParsed"
                )
                break
            except Exception as exc:  # pragma: no cover - network errors
                attempts += 1
                if attempts >= 5:
                    logger.error("On-chain scan failed: %s", exc)
                    return []
                logger.warning(
                    "RPC error: %s. Sleeping %s seconds before retry", exc, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    tokens: List[str] | List[Dict[str, Any]] = []
    for acc in resp.get("result", []):
        info = (
            acc.get("account", {})
            .get("data", {})
            .get("parsed", {})
            .get("info", {})
        )
        name = info.get("name", "")
        mint = info.get("mint")
        if name and name.lower().endswith("bonk"):
            volume = await asyncio.to_thread(
                onchain_metrics.fetch_volume_onchain, mint, rpc_url
            )
            liquidity = await asyncio.to_thread(
                onchain_metrics.fetch_liquidity_onchain, mint, rpc_url
            )
            if return_metrics:
                tokens.append({"address": mint, "volume": volume, "liquidity": liquidity})
            else:
                tokens.append(mint)
    logger.info("Found %d candidate on-chain tokens", len(tokens))
    return tokens


def scan_tokens_onchain_sync(rpc_url: str, *, return_metrics: bool = False) -> List[str] | List[Dict[str, Any]]:
    """Synchronous wrapper for :func:`scan_tokens_onchain`."""

    return asyncio.run(scan_tokens_onchain(rpc_url, return_metrics=return_metrics))


def fetch_mempool_tx_rate(token: str, rpc_url: str, limit: int = 20) -> float:
    """Return approximate mempool transaction rate for ``token``."""

    if not rpc_url:
        raise ValueError("rpc_url is required")

    cache_key = (token, rpc_url)
    cached = MEMPOOL_RATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    client = Client(rpc_url)
    try:
        resp = client.get_signatures_for_address(PublicKey(token), limit=limit)
        entries = resp.get("result", [])
        times = [e.get("blockTime") for e in entries if e.get("blockTime")]
        if len(times) >= 2:
            duration = max(times) - min(times)
            if duration > 0:
                rate = float(len(times)) / float(duration)
            else:
                rate = float(len(times))
        else:
            rate = float(len(times))
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch mempool rate for %s: %s", token, exc)
        rate = 0.0

    # update history for optional forecasting
    features = []
    try:
        from . import onchain_metrics  # circular safe

        depth_change = onchain_metrics.order_book_depth_change(token)
        whale = fetch_whale_wallet_activity(token, rpc_url)
        avg_swap = fetch_average_swap_size(token, rpc_url)
        features = [float(depth_change), float(rate), float(whale), float(avg_swap)]
    except Exception:
        features = [0.0, float(rate), 0.0, 0.0]

    hist = MEMPOOL_FEATURE_HISTORY.setdefault(cache_key, [])
    hist.append(features)
    model_path = os.getenv("ONCHAIN_MODEL_PATH")
    if model_path:
        from .models.onchain_forecaster import get_model

        model = get_model(model_path)
        if model is not None:
            seq_len = getattr(model, "seq_len", 30)
            if len(hist) >= seq_len:
                seq = hist[-seq_len:]
                try:
                    rate = float(model.predict(seq))
                except Exception as exc:  # pragma: no cover - model errors
                    logger.warning("Forecast failed: %s", exc)
    hist[:] = hist[-30:]

    MEMPOOL_RATE_CACHE.set(cache_key, rate)
    return rate


def fetch_whale_wallet_activity(
    token: str, rpc_url: str, threshold: float = 1_000_000.0
) -> float:
    """Return fraction of liquidity held by large accounts."""

    if not rpc_url:
        raise ValueError("rpc_url is required")

    cache_key = (token, rpc_url)
    cached = WHALE_ACTIVITY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    client = Client(rpc_url)
    try:
        resp = client.get_token_largest_accounts(PublicKey(token))
        accounts = resp.get("result", {}).get("value", [])
        total = 0.0
        whales = 0.0
        for acc in accounts:
            val = acc.get("uiAmount", acc.get("amount", 0))
            try:
                bal = float(val)
            except Exception:
                bal = 0.0
            total += bal
            if bal >= threshold:
                whales += bal
        activity = whales / total if total else 0.0
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch whale activity for %s: %s", token, exc)
        activity = 0.0

    WHALE_ACTIVITY_CACHE.set(cache_key, activity)
    return activity


def fetch_average_swap_size(token: str, rpc_url: str, limit: int = 20) -> float:
    """Return the average swap size for ``token`` based on recent signatures."""

    if not rpc_url:
        raise ValueError("rpc_url is required")

    cache_key = (token, rpc_url)
    cached = AVG_SWAP_SIZE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    client = Client(rpc_url)
    try:
        resp = client.get_signatures_for_address(PublicKey(token), limit=limit)
        entries = resp.get("result", [])
        total = 0.0
        count = 0
        for e in entries:
            amt = e.get("amount", 0.0)
            try:
                total += float(amt)
            except Exception:
                continue
            count += 1
        size = total / float(count) if count else 0.0
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch swap size for %s: %s", token, exc)
        size = 0.0

    AVG_SWAP_SIZE_CACHE.set(cache_key, size)
    return size
