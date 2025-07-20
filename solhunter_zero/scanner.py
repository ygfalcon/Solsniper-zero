from __future__ import annotations

import asyncio
import logging
import time
from typing import List

import requests

from . import scanner_common, dex_scanner

from .scanner_common import (
    BIRDEYE_API,
    HEADERS,
    OFFLINE_TOKENS,
    SOLANA_RPC_URL,
    fetch_trending_tokens,
    fetch_trending_tokens_async,
    fetch_raydium_listings,
    fetch_raydium_listings_async,
    fetch_orca_listings,
    fetch_orca_listings_async,
    offline_or_onchain,
    parse_birdeye_tokens,
    scan_tokens_from_file,
)


from .scanner_onchain import scan_tokens_onchain


logger = logging.getLogger(__name__)


def scan_tokens_from_pools() -> List[str]:
    """Public wrapper for pool discovery used by tests."""

    return scanner_common.scan_tokens_from_pools()



def scan_tokens(
    *, offline: bool = False, token_file: str | None = None, method: str = "websocket"
) -> List[str]:
    """Scan the Solana network for new tokens using ``method``."""
    if method == "websocket":
        tokens = offline_or_onchain(offline, token_file)
        if tokens is None:
            backoff = 1
            max_backoff = 60
            while True:
                try:
                    resp = requests.get(BIRDEYE_API, headers=HEADERS, timeout=10)
                    if resp.status_code == 429:
                        logger.warning("Rate limited (429). Sleeping %s seconds", backoff)
                        time.sleep(backoff)
                        backoff = min(backoff * 2, max_backoff)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    tokens = parse_birdeye_tokens(data)
                    backoff = 1
                    break
                except requests.RequestException as e:
                    logger.error("Scan failed: %s", e)
                    tokens = []
                    break
    elif offline:
        logger.info("Offline mode enabled, returning static tokens")
        tokens = OFFLINE_TOKENS
    elif method == "onchain":
        tokens = scan_tokens_onchain(scanner_common.SOLANA_RPC_URL)
    elif method == "pools":
        tokens = scan_tokens_from_pools()
    elif method == "file":
        tokens = scan_tokens_from_file()
    else:
        raise ValueError(f"unknown discovery method: {method}")

    if not offline and token_file is None:
        extra = fetch_trending_tokens()
        extra += fetch_raydium_listings()
        extra += fetch_orca_listings()
        tokens = list(dict.fromkeys(tokens + extra))
    return tokens




async def scan_tokens_async(
    *, offline: bool = False, token_file: str | None = None, method: str = "websocket"
) -> List[str]:

    """Async wrapper around :func:`scan_tokens` using aiohttp."""

    if method == "websocket":
        from .async_scanner import scan_tokens_async as _scan
        tokens = await _scan(offline=offline, token_file=token_file)
    elif offline:
        logger.info("Offline mode enabled, returning static tokens")
        tokens = OFFLINE_TOKENS
    elif method == "onchain":
        tokens = await asyncio.to_thread(
            scan_tokens_onchain, scanner_common.SOLANA_RPC_URL
        )
    elif method == "pools":
        tokens = await asyncio.to_thread(scan_tokens_from_pools)
    elif method == "file":
        tokens = await asyncio.to_thread(scan_tokens_from_file)
    else:
        raise ValueError(f"unknown discovery method: {method}")

    if not offline and token_file is None:
        extra = await fetch_trending_tokens_async()
        extra += await fetch_raydium_listings_async()
        extra += await fetch_orca_listings_async()
        tokens = list(dict.fromkeys(tokens + extra))
    return tokens

