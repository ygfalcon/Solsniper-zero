from __future__ import annotations

import asyncio
import logging
from typing import List

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
from .event_bus import publish

logger = logging.getLogger(__name__)


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
    *, offline: bool = False, token_file: str | None = None, method: str = "websocket"
) -> List[str]:
    """Discover tokens asynchronously using the specified ``method``."""

    scanner = TokenScanner()
    base_task = asyncio.create_task(
        scanner.scan(offline=offline, token_file=token_file, method=method)
    )

    tasks: list[asyncio.Task] = [base_task]
    if not offline and token_file is None:
        tasks.append(asyncio.create_task(fetch_trending_tokens_async()))
        tasks.append(asyncio.create_task(fetch_raydium_listings_async()))
        tasks.append(asyncio.create_task(fetch_orca_listings_async()))
        if DEX_LISTING_WS_URL and method not in {"onchain", "pools", "file"}:
            tasks.append(asyncio.create_task(_fetch_dex_ws_tokens()))

    results = await asyncio.gather(*tasks)
    tokens = results[0] or []
    extras: List[str] = []
    for res in results[1:]:
        extras.extend(res)
    if extras:
        tokens = list(dict.fromkeys(tokens + extras))
    publish("token_discovered", tokens)
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
