from __future__ import annotations

import asyncio
import logging
import time
from typing import List
import requests

import requests



from . import scanner_common, dex_scanner

from .scanner_common import (
    BIRDEYE_API,
    HEADERS,
    OFFLINE_TOKENS,
    SOLANA_RPC_URL,
    offline_or_onchain,
    parse_birdeye_tokens,
    scan_tokens_from_file,

    offline_or_onchain,
    SOLANA_RPC_URL,
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
        if tokens is not None:

            return tokens

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
                return tokens
            except requests.RequestException as e:
                logger.error("Scan failed: %s", e)
                return []


    if offline:
        logger.info("Offline mode enabled, returning static tokens")
        return OFFLINE_TOKENS

    if method == "onchain":
        return scan_tokens_onchain(scanner_common.SOLANA_RPC_URL)
    if method == "pools":
        return scan_tokens_from_pools()
    if method == "file":
        return scan_tokens_from_file()

    raise ValueError(f"unknown discovery method: {method}")




async def scan_tokens_async(
    *, offline: bool = False, token_file: str | None = None, method: str = "websocket"
) -> List[str]:

    """Async wrapper around :func:`scan_tokens` using aiohttp."""

    if method == "websocket":
        from .async_scanner import scan_tokens_async as _scan
        return await _scan(offline=offline, token_file=token_file)


    if offline:
        logger.info("Offline mode enabled, returning static tokens")
        return OFFLINE_TOKENS

    if method == "onchain":

        return await asyncio.to_thread(scan_tokens_onchain, scanner_common.SOLANA_RPC_URL)

    if method == "pools":
        return await asyncio.to_thread(scan_tokens_from_pools)
    if method == "file":
        return await asyncio.to_thread(scan_tokens_from_file)


    raise ValueError(f"unknown discovery method: {method}")

