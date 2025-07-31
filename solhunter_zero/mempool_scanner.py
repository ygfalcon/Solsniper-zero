from __future__ import annotations

import asyncio
import logging
import re
import os
import statistics
import contextlib
import psutil
from collections import deque
from typing import AsyncGenerator, Iterable, Dict, Any, Deque

from .event_bus import subscription

try:
    from solana.publickey import PublicKey
except Exception:  # pragma: no cover - minimal stub when solana is missing

    class PublicKey(str):
        def __new__(cls, value: str):
            obj = str.__new__(cls, value)
            try:
                from solders.pubkey import Pubkey

                obj._solder = Pubkey.from_string(value)
            except Exception:
                obj._solder = value
            return obj

        @property
        def _key(self):  # mimic solana-py attribute
            return getattr(self, "_solder", self)

    import types, sys

    mod = types.ModuleType("solana.publickey")
    mod.PublicKey = PublicKey
    sys.modules.setdefault("solana.publickey", mod)

from solana.rpc.websocket_api import RpcTransactionLogsFilterMentions, connect

from .scanner_onchain import TOKEN_PROGRAM_ID
from .dex_scanner import DEX_PROGRAM_ID
from .scanner_common import TOKEN_SUFFIX, TOKEN_KEYWORDS, token_matches
from . import onchain_metrics
from . import scanner_onchain
from . import order_book_ws

logger = logging.getLogger(__name__)

MEMPOOL_STATS_WINDOW = int(os.getenv("MEMPOOL_STATS_WINDOW", "5") or 5)
MEMPOOL_SCORE_THRESHOLD = float(os.getenv("MEMPOOL_SCORE_THRESHOLD", "0") or 0.0)

_ROLLING_STATS: Dict[str, Dict[str, Deque[float]]] = {}
_CPU_PERCENT: float = 0.0
_DYN_INTERVAL: float = 2.0

def _on_system_metrics(msg: Any) -> None:
    """Update :data:`_CPU_PERCENT` from a ``system_metrics`` event."""
    cpu = getattr(msg, "cpu", None)
    if isinstance(msg, dict):
        cpu = msg.get("cpu", cpu)
    if cpu is None:
        return
    try:
        global _CPU_PERCENT
        _CPU_PERCENT = float(cpu)
    except Exception:
        pass

_resource_sub = subscription("system_metrics", _on_system_metrics)
_resource_sub.__enter__()

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
                    if (
                        name
                        and mint
                        and token_matches(mint, name, suffix=suffix, keywords=keywords)
                    ):
                        tokens.add(mint)

                if include_pools:
                    for line in logs:
                        m = POOL_TOKEN_RE.search(line)
                        if m:
                            tok = m.group(1)
                            if token_matches(
                                tok, None, suffix=suffix, keywords=keywords
                            ):
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

    volume, liquidity, insights = await asyncio.gather(
        onchain_metrics.fetch_volume_onchain_async(token, rpc_url),
        onchain_metrics.fetch_liquidity_onchain_async(token, rpc_url),
        onchain_metrics.collect_onchain_insights_async(token, rpc_url),
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
    max_concurrency: int | None = None,
    cpu_usage_threshold: float | None = None,
    dynamic_concurrency: bool = False,
) -> AsyncGenerator[Dict[str, float], None]:
    """Yield ranked token events from the mempool."""

    if threshold is None:
        threshold = MEMPOOL_SCORE_THRESHOLD

    if max_concurrency is None or max_concurrency <= 0:
        cpu_count = os.cpu_count() or 1
        max_concurrency = max(1, cpu_count // 2)

    sem = asyncio.Semaphore(max_concurrency)
    current_limit = max_concurrency
    _dyn_interval = float(os.getenv("DYNAMIC_CONCURRENCY_INTERVAL", str(_DYN_INTERVAL)) or _DYN_INTERVAL)
    high = float(os.getenv("CPU_HIGH_THRESHOLD", "80") or 80)
    low = float(os.getenv("CPU_LOW_THRESHOLD", "40") or 40)
    adjust_task: asyncio.Task | None = None

    async def _set_limit(new_limit: int) -> None:
        nonlocal current_limit
        diff = new_limit - current_limit
        if diff > 0:
            for _ in range(diff):
                sem.release()
        elif diff < 0:
            for _ in range(-diff):
                await sem.acquire()
        current_limit = new_limit

    if dynamic_concurrency:
        async def _adjust() -> None:
            nonlocal current_limit
            try:
                while True:
                    await asyncio.sleep(_dyn_interval)
                    cpu = psutil.cpu_percent()
                    if cpu > high and current_limit == max_concurrency:
                        await _set_limit(max(1, max_concurrency // 2))
                    elif cpu < low and current_limit < max_concurrency:
                        await _set_limit(max_concurrency)
            except asyncio.CancelledError:
                pass

        adjust_task = asyncio.create_task(_adjust())
    elif cpu_usage_threshold is not None:
        async def _adjust_threshold() -> None:
            nonlocal current_limit
            try:
                while True:
                    await asyncio.sleep(_dyn_interval)
                    cpu = psutil.cpu_percent()
                    if cpu > cpu_usage_threshold and current_limit == max_concurrency:
                        await _set_limit(max(1, max_concurrency // 2))
                    elif cpu <= cpu_usage_threshold and current_limit < max_concurrency:
                        await _set_limit(max_concurrency)
            except asyncio.CancelledError:
                pass

        adjust_task = asyncio.create_task(_adjust_threshold())
    queue: asyncio.Queue[Dict[str, float]] = asyncio.Queue()
    tasks: set[asyncio.Task] = set()

    async def worker(addr: str) -> None:
        async with sem:
            score, data = await rank_token(addr, rpc_url)
            if score >= threshold:
                combined = data["momentum"] * (1.0 - data["whale_activity"])
                await queue.put({"address": addr, **data, "combined_score": combined})

    async for tok in stream_mempool_tokens(
        rpc_url,
        suffix=suffix,
        keywords=keywords,
        include_pools=include_pools,
    ):
        address = tok["address"] if isinstance(tok, dict) else tok
        task = asyncio.create_task(worker(address))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        while not queue.empty():
            yield queue.get_nowait()

    if tasks:
        await asyncio.gather(*tasks)
    while not queue.empty():
        yield queue.get_nowait()
    if adjust_task:
        adjust_task.cancel()
        with contextlib.suppress(Exception):
            await adjust_task


async def stream_ranked_mempool_tokens_with_depth(
    rpc_url: str,
    *,
    depth_threshold: float = 0.0,
    **kwargs,
) -> AsyncGenerator[Dict[str, float], None]:
    """Yield ranked mempool tokens enriched with depth metrics."""

    async for event in stream_ranked_mempool_tokens(rpc_url, **kwargs):
        token = event["address"]
        depth, _imb, txr = order_book_ws.snapshot(token)
        event["depth"] = depth
        event["depth_tx_rate"] = txr
        event["combined_score"] += depth + txr
        if depth >= depth_threshold:
            yield event
