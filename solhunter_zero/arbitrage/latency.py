"""Latency measurement utilities for DEX endpoints."""
import asyncio
import contextlib
import logging
import os
import time
from typing import Mapping

import aiohttp

from solhunter_zero.lru import TTLCache

from ..system import detect_cpu_count
from ..http import get_session
from ..dynamic_limit import _target_concurrency, _step_limit
from .. import resource_monitor
from ..event_bus import publish
from ..scanner_common import JUPITER_WS_URL
from .routing import (
    VENUE_URLS,
    EXTRA_API_URLS,
    EXTRA_WS_URLS,
    DEX_LATENCY,
    _routeffi,
)

logger = logging.getLogger(__name__)

# Websocket endpoints for latency checks
ORCA_WS_URL = os.getenv("ORCA_WS_URL", "")
RAYDIUM_WS_URL = os.getenv("RAYDIUM_WS_URL", "")
PHOENIX_WS_URL = os.getenv("PHOENIX_WS_URL", "")
METEORA_WS_URL = os.getenv("METEORA_WS_URL", "")

# Measure DEX latency on import unless disabled via environment variable
MEASURE_DEX_LATENCY = os.getenv("MEASURE_DEX_LATENCY", "1").lower() not in {
    "0",
    "false",
    "no",
}

# Interval in seconds between latency refreshes
DEX_LATENCY_REFRESH_INTERVAL = float(
    os.getenv("DEX_LATENCY_REFRESH_INTERVAL", "60") or 60
)

# Interval in seconds between dynamic concurrency adjustments
_DYN_INTERVAL: float = 2.0

DEX_LATENCY_CACHE_TTL = float(os.getenv("DEX_LATENCY_CACHE_TTL", "30") or 30)
DEX_LATENCY_CACHE = TTLCache(maxsize=64, ttl=DEX_LATENCY_CACHE_TTL)

_LATENCY_TASK: asyncio.Task | None = None


async def _ping_url(
    session: aiohttp.ClientSession, url: str, attempts: int = 3
) -> float:
    """Return the average latency for ``url`` in seconds."""

    async def _once() -> float | None:
        start = time.perf_counter()
        try:
            if url.startswith("ws"):
                async with session.ws_connect(url, timeout=5):
                    pass
            else:
                async with session.get(url, timeout=5) as resp:
                    await resp.read()
        except Exception:  # pragma: no cover - network failures
            return None
        return time.perf_counter() - start

    coros = [_once() for _ in range(max(1, attempts))]
    results = await asyncio.gather(*coros, return_exceptions=True)
    times = [t for t in results if isinstance(t, (int, float))]
    if times:
        return sum(times) / len(times)
    return 0.0


async def measure_dex_latency_async(
    urls: Mapping[str, str],
    attempts: int = 3,
    *,
    max_concurrency: int | None = None,
    dynamic_concurrency: bool = False,
) -> dict[str, float]:
    """Asynchronously measure latency for each URL in ``urls``."""

    cache_key = (tuple(sorted(urls.items())), attempts)
    cached = DEX_LATENCY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if _routeffi.available():
        func = getattr(_routeffi, "measure_latency", None)
        if callable(func):
            try:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(
                    None, lambda: func(dict(urls), attempts)
                )
                if res:
                    DEX_LATENCY_CACHE.set(cache_key, res)
                    return res
            except Exception as exc:  # pragma: no cover - optional ffi failures
                logger.debug("ffi.measure_latency failed: %s", exc)

    if max_concurrency is None or max_concurrency <= 0:
        max_concurrency = max(detect_cpu_count(), len(urls))

    class _Noop:
        async def __aenter__(self):
            return None

        async def __aexit__(self, exc_type, exc, tb):
            pass

    use_sem = dynamic_concurrency or max_concurrency < len(urls)
    sem: asyncio.Semaphore | _Noop
    if use_sem:
        sem = asyncio.Semaphore(max_concurrency)
    else:
        sem = _Noop()
    current_limit = max_concurrency
    _dyn_interval = float(
        os.getenv("DYNAMIC_CONCURRENCY_INTERVAL", str(_DYN_INTERVAL)) or _DYN_INTERVAL
    )
    ewm = float(os.getenv("CONCURRENCY_EWM_SMOOTHING", "0.15") or 0.15)
    kp = float(
        os.getenv("CONCURRENCY_SMOOTHING", os.getenv("CONCURRENCY_KP", "0.5")) or 0.5
    )
    ki = float(os.getenv("CONCURRENCY_KI", "0.0") or 0.0)
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
                    cpu = resource_monitor.get_cpu_usage()
                    target = _target_concurrency(
                        cpu, max_concurrency, low, high, smoothing=ewm
                    )
                    new_limit = _step_limit(
                        current_limit,
                        target,
                        max_concurrency,
                        smoothing=kp,
                        ki=ki,
                    )
                    if new_limit != current_limit:
                        await _set_limit(new_limit)
            except asyncio.CancelledError:
                pass

        adjust_task = asyncio.create_task(_adjust())
        await asyncio.sleep(0)

    session = await get_session()

    async def _measure(name: str, url: str) -> tuple[str, float]:
        async with sem:
            value = await _ping_url(session, url, attempts)
        return name, value

    coros = [_measure(n, u) for n, u in urls.items() if u]
    results = await asyncio.gather(*coros, return_exceptions=True)
    latency = {}
    for res in results:
        if isinstance(res, tuple):
            n, v = res
            latency[n] = v
    if adjust_task:
        adjust_task.cancel()
        with contextlib.suppress(Exception):
            await adjust_task
    DEX_LATENCY_CACHE.set(cache_key, latency)
    return latency


def measure_dex_latency(
    urls: Mapping[str, str] | None = None, attempts: int = 3
) -> dict[str, float]:
    """Synchronously measure latency for DEX endpoints."""

    if urls is None:
        urls = {**VENUE_URLS, **EXTRA_API_URLS, **EXTRA_WS_URLS}
        ws_map = {
            "orca": ORCA_WS_URL,
            "raydium": RAYDIUM_WS_URL,
            "phoenix": PHOENIX_WS_URL,
            "meteora": METEORA_WS_URL,
            "jupiter": JUPITER_WS_URL,
        }
        for name, url in ws_map.items():
            if url:
                urls.setdefault(name, url)

    return asyncio.run(measure_dex_latency_async(urls, attempts))


if MEASURE_DEX_LATENCY:
    try:
        DEX_LATENCY.update(measure_dex_latency())
    except Exception as exc:  # pragma: no cover - measurement failures
        logger.debug("DEX latency measurement failed: %s", exc)


async def _latency_loop(interval: float) -> None:
    """Background latency measurement loop."""
    try:
        while True:
            res = await measure_dex_latency_async(VENUE_URLS, dynamic_concurrency=True)
            if res:
                DEX_LATENCY.update(res)
                publish("dex_latency_update", res)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:  # pragma: no cover - cancellation
        pass


def start_latency_refresh(
    interval: float = DEX_LATENCY_REFRESH_INTERVAL,
) -> asyncio.Task:
    """Start periodic DEX latency refresh task."""
    global _LATENCY_TASK
    if _LATENCY_TASK is None or _LATENCY_TASK.done():
        loop = asyncio.get_running_loop()
        _LATENCY_TASK = loop.create_task(_latency_loop(interval))
    return _LATENCY_TASK


def stop_latency_refresh() -> None:
    """Cancel the running latency refresh task, if any."""
    global _LATENCY_TASK
    if _LATENCY_TASK is not None:
        _LATENCY_TASK.cancel()
        _LATENCY_TASK = None
