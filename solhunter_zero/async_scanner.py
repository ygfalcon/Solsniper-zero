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


async def scan_tokens_async(

    *, offline: bool = False, token_file: str | None = None, method: str = "websocket",

) -> List[str]:
    """Async variant of :func:`scanner.scan_tokens` with concurrent scanning."""
    base_task = asyncio.create_task(
        offline_or_onchain_async(offline, token_file, method=method)
    )
    birdeye_task = None
    if (
        not offline
        and token_file is None
        and method not in {"onchain", "pools", "file"}
        and scanner_common.BIRDEYE_API_KEY
    ):
        birdeye_task = asyncio.create_task(_fetch_birdeye_tokens())
    extra_tasks: list[asyncio.Task] = []
    if not offline and token_file is None:
        extra_tasks.append(asyncio.create_task(fetch_trending_tokens_async()))
        extra_tasks.append(asyncio.create_task(fetch_raydium_listings_async()))
        extra_tasks.append(asyncio.create_task(fetch_orca_listings_async()))
    results = await asyncio.gather(
        base_task,
        *(t for t in (birdeye_task,) if t),
        *extra_tasks,
    )
    idx = 0
    base_tokens = results[idx]
    idx += 1
    birdeye_tokens = []
    if birdeye_task:
        birdeye_tokens = results[idx]
        idx += 1
    extras = []
    for res in results[idx:]:
        extras.extend(res)
    tokens = base_tokens if base_tokens is not None else birdeye_tokens
    if not offline and token_file is None:
        tokens = list(dict.fromkeys((tokens or []) + extras))
    else:
        tokens = tokens or []
    return tokens
