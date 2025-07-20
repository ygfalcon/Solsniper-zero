from __future__ import annotations
import asyncio
import logging
from typing import List

import aiohttp

from .scanner_common import (
    BIRDEYE_API,
    HEADERS,
    fetch_trending_tokens_async,
    fetch_raydium_listings_async,
    fetch_orca_listings_async,
    offline_or_onchain_async,
    parse_birdeye_tokens,
)
from . import scanner_common
from . import dex_ws
from . import scanner_onchain

logger = logging.getLogger(__name__)


async def _fetch_birdeye_tokens() -> List[str]:
    """Fetch token addresses from the BirdEye API asynchronously."""
    backoff = 1
    max_backoff = 60
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(BIRDEYE_API, headers=HEADERS, timeout=10) as resp:
                    if resp.status == 429:
                        logger.warning("Rate limited (429). Sleeping %s seconds", backoff)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, max_backoff)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    return parse_birdeye_tokens(data)
        except aiohttp.ClientError as exc:
            logger.error("Scan failed: %s", exc)
            return []


async def _fetch_dex_ws_tokens() -> List[str]:
    """Return tokens from the DEX listing websocket if configured."""
    url = scanner_common.DEX_LISTING_WS_URL
    if not url:
        return []

    gen = dex_ws.stream_listed_tokens(url)
    tokens: List[str] = []
    try:
        while True:
            tokens.append(await asyncio.wait_for(anext(gen), timeout=0.1))
    except (StopAsyncIteration, asyncio.TimeoutError):  # pragma: no cover - no data
        pass
    finally:
        await gen.aclose()
    return tokens


async def scan_tokens_async(

    *, offline: bool = False, token_file: str | None = None, method: str = "websocket",

) -> List[str]:
    """Async variant of :func:`scanner.scan_tokens` with concurrent scanning."""
    use_metrics = not scanner_common.BIRDEYE_API_KEY
    if (
        use_metrics
        and not offline
        and token_file is None
        and method == "onchain"
    ):
        base_task = asyncio.create_task(
            asyncio.to_thread(
                scanner_onchain.scan_tokens_onchain,
                scanner_common.SOLANA_RPC_URL,
                return_metrics=True,
            )
        )
    elif (
        use_metrics
        and not offline
        and token_file is None
        and method == "mempool"
    ):
        from .mempool_scanner import stream_mempool_tokens

        async def _mempool():
            gen = stream_mempool_tokens(
                scanner_common.SOLANA_RPC_URL, return_metrics=True
            )
            tokens = []
            try:
                tokens.append(await anext(gen))
                try:
                    tokens.append(await anext(gen))
                except StopAsyncIteration:
                    pass
            except StopAsyncIteration:
                tokens = []
            finally:
                await gen.aclose()
            return tokens

        base_task = asyncio.create_task(_mempool())
    else:
        base_task = asyncio.create_task(
            offline_or_onchain_async(offline, token_file, method=method)
        )
    birdeye_task = None
    dex_ws_task = None
    if (
        not offline
        and token_file is None
        and method not in {"onchain", "pools", "file"}
        and scanner_common.BIRDEYE_API_KEY
    ):
        birdeye_task = asyncio.create_task(_fetch_birdeye_tokens())
    if (
        not offline
        and token_file is None
        and method not in {"onchain", "pools", "file"}
        and scanner_common.DEX_LISTING_WS_URL
    ):
        dex_ws_task = asyncio.create_task(_fetch_dex_ws_tokens())
    extra_tasks: list[asyncio.Task] = []
    if not offline and token_file is None:
        extra_tasks.append(asyncio.create_task(fetch_trending_tokens_async()))
        extra_tasks.append(asyncio.create_task(fetch_raydium_listings_async()))
        extra_tasks.append(asyncio.create_task(fetch_orca_listings_async()))
    results = await asyncio.gather(
        base_task,
        *(t for t in (birdeye_task, dex_ws_task) if t),
        *extra_tasks,
    )
    idx = 0
    base_data = results[idx]
    idx += 1
    birdeye_tokens = []
    if birdeye_task:
        birdeye_tokens = results[idx]
        idx += 1
    dex_ws_tokens = []
    if dex_ws_task:
        dex_ws_tokens = results[idx]
        idx += 1
    extras = []
    for res in results[idx:]:
        extras.extend(res)

    if use_metrics and not offline and token_file is None and method in {"onchain", "mempool"}:
        base_tokens = [e["address"] for e in (base_data or [])]
    else:
        base_tokens = base_data

    tokens = base_tokens if base_tokens is not None else birdeye_tokens
    if not offline and token_file is None:
        tokens = list(dict.fromkeys((tokens or []) + dex_ws_tokens + extras))
    else:
        tokens = tokens or []
    return tokens
