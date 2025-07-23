from __future__ import annotations

import asyncio
import logging
import re
import os
import statistics
from collections import deque
from typing import AsyncGenerator, Iterable, Dict, Any, Deque

from solana.publickey import PublicKey
from solana.rpc.websocket_api import RpcTransactionLogsFilterMentions, connect

from .scanner_onchain import TOKEN_PROGRAM_ID
from .dex_scanner import DEX_PROGRAM_ID
from .scanner_common import TOKEN_SUFFIX, TOKEN_KEYWORDS, token_matches
from . import onchain_metrics
from . import scanner_onchain

logger = logging.getLogger(__name__)

MEMPOOL_STATS_WINDOW = int(os.getenv("MEMPOOL_STATS_WINDOW", "5") or 5)
MEMPOOL_SCORE_THRESHOLD = float(os.getenv("MEMPOOL_SCORE_THRESHOLD", "0") or 0.0)

_ROLLING_STATS: Dict[str, Dict[str, Deque[float]]] = {}

NAME_RE = re.compile(r"name:\s*(\S+)", re.IGNORECASE)
MINT_RE = re.compile(r"mint:\s*(\S+)", re.IGNORECASE)
POOL_TOKEN_RE = re.compile(r"token[AB]:\s*([A-Za-z0-9]{32,44})", re.IGNORECASE)

async def stream_mempool_tokens(
    rpc_url: str,
    *,
    suffix: str | None = None,
    keywords: Iterable[str] | None = None,
    include_pools: bool = True,
    return_metrics: bool = False,
) -> AsyncGenerator[str | Dict[str, Any], None]:
    """Yield token mints from unconfirmed transactions."""

    if not rpc_url:
        if False:
            yield None
        return

    if suffix is None:
        suffix = TOKEN_SUFFIX
    if keywords is None:
        keywords = TOKEN_KEYWORDS
    suffix = suffix.lower() if suffix else None

    async with connect(rpc_url) as ws:
        await ws.logs_subscribe(
            RpcTransactionLogsFilterMentions(PublicKey(str(TOKEN_PROGRAM_ID))._key),
            commitment="processed",
        )
        if include_pools:
            await ws.logs_subscribe(
                RpcTransactionLogsFilterMentions(PublicKey(str(DEX_PROGRAM_ID))._key),
                commitment="processed",
            )

        while True:
            try:
                msgs = await ws.recv()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover - network errors
                logger.error("Websocket error: %s", exc)
                await asyncio.sleep(1)
                continue

            for msg in msgs:
                try:
                    logs = msg.result.value.logs  # type: ignore[attr-defined]
                except Exception:
                    try:
                        logs = msg["result"]["value"]["logs"]  # type: ignore[index]
                    except Exception:
                        continue

                tokens = set()
                if any("InitializeMint" in l for l in logs):
                    name = None
                    mint = None
                    for line in logs:
                        if name is None:
                            m = NAME_RE.search(line)
                            if m:
                                name = m.group(1)
                        if mint is None:
                            m = MINT_RE.search(line)
                            if m:
                                mint = m.group(1)
                    if name and mint and token_matches(mint, name, suffix=suffix, keywords=keywords):
                        tokens.add(mint)

                if include_pools:
                    for line in logs:
                        m = POOL_TOKEN_RE.search(line)
                        if m:
                            tok = m.group(1)
                            if token_matches(tok, None, suffix=suffix, keywords=keywords):
                                tokens.add(tok)

                for tok in tokens:
                    volume = 0.0
                    liquidity = 0.0
                    if return_metrics:
                        volume = await asyncio.to_thread(
                            onchain_metrics.fetch_volume_onchain, tok, rpc_url
                        )
                        liquidity = await asyncio.to_thread(
                            onchain_metrics.fetch_liquidity_onchain, tok, rpc_url
                        )
                        yield {"address": tok, "volume": volume, "liquidity": liquidity}
                    else:
                        yield tok


async def rank_token(token: str, rpc_url: str) -> tuple[float, Dict[str, float]]:
    """Return ranking score and metrics for ``token``."""

    volume = await asyncio.to_thread(
        onchain_metrics.fetch_volume_onchain, token, rpc_url
    )
    liquidity = await asyncio.to_thread(
        onchain_metrics.fetch_liquidity_onchain, token, rpc_url
    )
    insights = await asyncio.to_thread(
        onchain_metrics.collect_onchain_insights, token, rpc_url
    )
    tx_rate = insights.get("tx_rate", 0.0)
    whale_activity = insights.get("whale_activity", 0.0)
    avg_swap = insights.get("avg_swap_size", 0.0)

    wallet_conc = 1.0 - float(whale_activity)

    def _update(token: str, key: str, value: float) -> Deque[float]:
        dq = _ROLLING_STATS.setdefault(token, {}).setdefault(
            key, deque(maxlen=MEMPOOL_STATS_WINDOW)
        )
        dq.append(float(value))
        return dq

    tx_hist = _update(token, "tx", tx_rate)
    _update(token, "wallet", wallet_conc)
    _update(token, "swap", avg_swap)

    momentum = 0.0
    anomaly = 0.0
    if len(tx_hist) > 1:
        prev_avg = sum(list(tx_hist)[:-1]) / (len(tx_hist) - 1)
        momentum = tx_hist[-1] - prev_avg
        if len(tx_hist) > 2:
            mean = statistics.mean(list(tx_hist)[:-1])
            stdev = statistics.stdev(list(tx_hist)[:-1]) or 1.0
            anomaly = (tx_hist[-1] - mean) / stdev

    score = (
        float(volume)
        + float(liquidity)
        + float(tx_rate)
        + momentum
        + anomaly
        - float(whale_activity)
    )
    metrics = {
        "volume": float(volume),
        "liquidity": float(liquidity),
        "tx_rate": float(tx_rate),
        "whale_activity": float(whale_activity),
        "wallet_concentration": wallet_conc,
        "avg_swap_size": float(avg_swap),
        "momentum": momentum,
        "anomaly": anomaly,
        "score": score,
    }
    return score, metrics


async def stream_ranked_mempool_tokens(
    rpc_url: str,
    *,
    suffix: str | None = None,
    keywords: Iterable[str] | None = None,
    include_pools: bool = True,
    threshold: float | None = None,
) -> AsyncGenerator[Dict[str, float], None]:
    """Yield ranked token events from the mempool."""

    if threshold is None:
        threshold = MEMPOOL_SCORE_THRESHOLD

    async for tok in stream_mempool_tokens(
        rpc_url,
        suffix=suffix,
        keywords=keywords,
        include_pools=include_pools,
    ):
        address = tok["address"] if isinstance(tok, dict) else tok
        score, data = await rank_token(address, rpc_url)
        if score >= threshold:
            yield {"address": address, **data}
