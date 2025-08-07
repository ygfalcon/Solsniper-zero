from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

from . import onchain_metrics
from .mempool_scanner import (
    stream_ranked_mempool_tokens,
    stream_ranked_mempool_tokens_with_depth,
)
from .scanner_common import fetch_trending_tokens_async
from .scanner_onchain import scan_tokens_onchain


async def _metrics_for(token: str, rpc_url: str) -> tuple[float, float]:
    """Return volume and liquidity for ``token`` via on-chain metrics."""

    volume = await asyncio.to_thread(
        onchain_metrics.fetch_volume_onchain, token, rpc_url
    )
    liquidity = await asyncio.to_thread(
        onchain_metrics.fetch_liquidity_onchain, token, rpc_url
    )
    return float(volume), float(liquidity)


async def merge_sources(
    rpc_url: str,
    *,
    mempool_limit: int = 10,
    limit: int | None = None,
    mempool_threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """Return ranked tokens collected from multiple discovery sources."""

    if mempool_threshold is None:
        mempool_threshold = float(os.getenv("MEMPOOL_SCORE_THRESHOLD", "0") or 0.0)

    trend_task = asyncio.create_task(fetch_trending_tokens_async())
    onchain_task = asyncio.create_task(
        scan_tokens_onchain(rpc_url, return_metrics=True)
    )

    mp_gen = stream_ranked_mempool_tokens_with_depth(
        rpc_url, threshold=mempool_threshold
    )
    mp_tokens: List[Dict[str, Any]] = []
    try:
        while len(mp_tokens) < mempool_limit:
            tok = await asyncio.wait_for(anext(mp_gen), timeout=0.5)
            mp_tokens.append(tok)
    except (StopAsyncIteration, asyncio.TimeoutError):
        pass
    finally:
        await mp_gen.aclose()

    trending = await trend_task
    onchain_tokens = await onchain_task

    trend_metrics = []
    if trending:
        vols_liqs = await asyncio.gather(*(_metrics_for(t, rpc_url) for t in trending))
        for addr, (vol, liq) in zip(trending, vols_liqs):
            trend_metrics.append({"address": addr, "volume": vol, "liquidity": liq})

    combined: Dict[str, Dict[str, float]] = {}

    def add_entry(entry: Dict[str, Any]) -> None:
        addr = entry.get("address")
        if not addr:
            return
        vol = float(entry.get("volume", 0.0))
        liq = float(entry.get("liquidity", 0.0))
        score = float(entry.get("score", 0.0))
        current = combined.setdefault(
            addr,
            {
                "address": addr,
                "volume": 0.0,
                "liquidity": 0.0,
                "score": 0.0,
            },
        )
        if vol > current["volume"]:
            current["volume"] = vol
        if liq > current["liquidity"]:
            current["liquidity"] = liq
        if score > current.get("score", 0.0):
            current["score"] = score
        for key in ("momentum", "anomaly", "wallet_concentration", "avg_swap_size"):
            val = entry.get(key)
            if val is not None:
                current[key] = float(val)

    for e in onchain_tokens:
        add_entry(e)
    for e in mp_tokens:
        add_entry(e)
    for e in trend_metrics:
        add_entry(e)

    result = list(combined.values())
    result.sort(
        key=lambda x: (
            x.get("score", 0.0),
            x.get("volume", 0.0) + x.get("liquidity", 0.0),
        ),
        reverse=True,
    )
    if limit is not None:
        result = result[:limit]
    return result
