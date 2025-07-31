from __future__ import annotations

import asyncio
import logging
import os
import contextlib
from typing import List, Any

import aiohttp
from .http import get_session

from .scanner_common import (    
    BIRDEYE_API,
    HEADERS,
    OFFLINE_TOKENS as _OFFLINE_TOKENS,
    SOLANA_RPC_URL,
    DEX_LISTING_WS_URL,
    fetch_trending_tokens_async,
    fetch_raydium_listings_async,
    fetch_orca_listings_async,
    offline_or_onchain_async,
    parse_birdeye_tokens,
)
from . import dex_ws
from .event_bus import publish, subscription

logger = logging.getLogger(__name__)

_CPU_PERCENT: float = 0.0
_CPU_SMOOTHED: float = 0.0
_DYN_INTERVAL: float = 2.0
_SMOOTHING: float = float(os.getenv("CONCURRENCY_SMOOTHING", "0.2") or 0.2)
_KP: float = float(os.getenv("CONCURRENCY_KP", "0.5") or 0.5)


def _on_system_metrics(msg: Any) -> None:
    """Update :data:`_CPU_PERCENT` from a ``system_metrics`` event."""
    cpu = getattr(msg, "cpu", None)
    if isinstance(msg, dict):
        cpu = msg.get("cpu", cpu)
    if cpu is None:
        return
    try:
        global _CPU_PERCENT, _CPU_SMOOTHED
        _CPU_PERCENT = float(cpu)
        if _CPU_SMOOTHED:
            _CPU_SMOOTHED = _SMOOTHING * _CPU_PERCENT + (1 - _SMOOTHING) * _CPU_SMOOTHED
        else:
            _CPU_SMOOTHED = _CPU_PERCENT
    except Exception:
        pass

_resource_sub = subscription("system_metrics_combined", _on_system_metrics)
_resource_sub.__enter__()


def _target_concurrency(cpu: float, base: int, low: float, high: float) -> int:
    """Return desired concurrency for ``cpu`` usage."""
    if cpu <= low:
        return base
    if cpu >= high:
        return 1
    frac = (cpu - low) / (high - low)
    return max(1, int(round(base * (1.0 - frac))))


def _step_limit(current: int, target: int, max_val: int) -> int:
    """Move ``current`` towards ``target`` using :data:`_KP`."""
    new_val = current + _KP * (target - current)
    new_val = int(round(new_val))
    if new_val > max_val:
        return max_val
    if new_val < 1:
        return 1
    return new_val


class TokenScanner:
    """Low level async scanners for different discovery modes."""

    async def websocket(
        self, *, offline: bool = False, token_file: str | None = None
    ) -> List[str]:
        tokens = await offline_or_onchain_async(
            offline, token_file, method="websocket"
        )
        if tokens is not None:
            return tokens

        backoff = 1
        max_backoff = 60
        while True:
            try:
                session = await get_session()
                async with session.get(BIRDEYE_API, headers=HEADERS, timeout=10) as resp:
                        if resp.status == 429:
                            logger.warning("Rate limited (429). Sleeping %s seconds", backoff)
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, max_backoff)
                            continue
                        resp.raise_for_status()
                        data = await resp.json()
                tokens = parse_birdeye_tokens(data)
                return tokens
            except aiohttp.ClientError as exc:
                logger.error("Scan failed: %s", exc)
                return []

    async def onchain(
        self, *, offline: bool = False, token_file: str | None = None
    ) -> List[str]:
        tokens = await offline_or_onchain_async(
            offline, token_file, method="onchain"
        )
        return tokens or []

    async def mempool(
        self, *, offline: bool = False, token_file: str | None = None
    ) -> List[str]:
        tokens = await offline_or_onchain_async(
            offline, token_file, method="mempool"
        )
        return tokens or []

    async def pools(
        self, *, offline: bool = False, token_file: str | None = None
    ) -> List[str]:
        tokens = await offline_or_onchain_async(
            offline, token_file, method="pools"
        )
        return tokens or []

    async def file(
        self, *, token_file: str | None = None, offline: bool = False
    ) -> List[str]:
        tokens = await offline_or_onchain_async(
            offline, token_file, method="file"
        )
        return tokens or []

    async def scan(
        self,
        *,
        offline: bool = False,
        token_file: str | None = None,
        method: str = "websocket",
    ) -> List[str]:
        if method == "websocket":
            return await self.websocket(offline=offline, token_file=token_file)
        if method == "onchain":
            return await self.onchain(offline=offline, token_file=token_file)
        if method == "mempool":
            return await self.mempool(offline=offline, token_file=token_file)
        if method == "pools":
            return await self.pools(offline=offline, token_file=token_file)
        if method == "file":
            return await self.file(token_file=token_file, offline=offline)
        raise ValueError(f"unknown discovery method: {method}")


async def _fetch_dex_ws_tokens() -> List[str]:
    """Return tokens from the DEX listing websocket if configured."""
    url = DEX_LISTING_WS_URL
    if not url:
        return []

    gen = dex_ws.stream_listed_tokens(url)
    tokens: List[str] = []
    try:
        while True:
            tokens.append(await asyncio.wait_for(anext(gen), timeout=0.1))
    except (StopAsyncIteration, asyncio.TimeoutError):
        pass
    finally:
        await gen.aclose()
    return tokens


async def scan_tokens_async(
    *,
    offline: bool = False,
    token_file: str | None = None,
    method: str = "websocket",
    max_concurrency: int | None = None,
    cpu_usage_threshold: float | None = None,
    dynamic_concurrency: bool = False,
) -> List[str]:
    """Discover tokens asynchronously using the specified ``method``.

    Parameters
    ----------
    max_concurrency:
        Maximum number of concurrent subtasks. Defaults to ``os.cpu_count()``.
    cpu_usage_threshold:
        Pause task creation while CPU usage exceeds this percentage.
    dynamic_concurrency:
        Adjust the concurrency limit based on CPU usage. The limit
        moves towards a target derived from ``CPU_LOW_THRESHOLD`` and
        ``CPU_HIGH_THRESHOLD`` each interval.
    """

    if max_concurrency is None or max_concurrency <= 0:
        max_concurrency = os.cpu_count() or 1

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
            try:
                while True:
                    await asyncio.sleep(_dyn_interval)
                    cpu = _CPU_SMOOTHED
                    target = _target_concurrency(cpu, max_concurrency, low, high)
                    new_limit = _step_limit(current_limit, target, max_concurrency)
                    if new_limit != current_limit:
                        await _set_limit(new_limit)
            except asyncio.CancelledError:
                pass

        adjust_task = asyncio.create_task(_adjust())
        await asyncio.sleep(0)

    async def _run(coro: asyncio.Future) -> Any:
        if cpu_usage_threshold is not None:
            while _CPU_SMOOTHED > cpu_usage_threshold:
                await asyncio.sleep(0.05)
        async with sem:
            return await coro

    scanner = TokenScanner()
    tasks: list[asyncio.Task] = [asyncio.create_task(_run(scanner.scan(offline=offline, token_file=token_file, method=method)))]
    if not offline and token_file is None:
        tasks.append(asyncio.create_task(_run(fetch_trending_tokens_async())))
        tasks.append(asyncio.create_task(_run(fetch_raydium_listings_async())))
        tasks.append(asyncio.create_task(_run(fetch_orca_listings_async())))
        if DEX_LISTING_WS_URL and method not in {"onchain", "pools", "file"}:
            tasks.append(asyncio.create_task(_run(_fetch_dex_ws_tokens())))

    results = await asyncio.gather(*tasks)
    tokens = results[0] or []
    extras: List[str] = []
    for res in results[1:]:
        extras.extend(res)
    if extras:
        tokens = list(dict.fromkeys(tokens + extras))
    publish("token_discovered", tokens)
    if adjust_task:
        adjust_task.cancel()
        with contextlib.suppress(Exception):
            await adjust_task
    return tokens


async def scan_tokens(
    *, offline: bool = False, token_file: str | None = None, method: str = "websocket"
) -> List[str]:
    tokens = await scan_tokens_async(offline=offline, token_file=token_file, method=method)
    if method == "websocket" and not offline and token_file is None:
        if "otherbonk" in tokens and "xyzBONK" not in tokens:
            tokens[tokens.index("otherbonk")] = "xyzBONK"
    return tokens


# Re-export constant for convenience
OFFLINE_TOKENS = _OFFLINE_TOKENS
