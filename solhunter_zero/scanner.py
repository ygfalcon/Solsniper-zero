from __future__ import annotations
import asyncio
import requests
import logging
import time
from typing import List

from .scanner_common import (
    BIRDEYE_API,
    HEADERS,
    OFFLINE_TOKENS,
    offline_or_onchain,
    parse_birdeye_tokens,
)

logger = logging.getLogger(__name__)


def scan_tokens(*, offline: bool = False, method: str = "birdeye") -> List[str]:
    """Scan the Solana network for new tokens ending with 'bonk'.

    When ``method`` is ``"pools"`` the DEX pool scanner is used instead of
    BirdEye.
    """
    if method == "pools":
        from . import dex_scanner
        from . import scanner_common
        return dex_scanner.scan_new_pools(scanner_common.SOLANA_RPC_URL)

    if method not in {"birdeye", "tokens"}:
        raise ValueError(f"unknown method: {method}")
    tokens = offline_or_onchain(offline)
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


async def scan_tokens_async(*, offline: bool = False, method: str = "birdeye") -> List[str]:
    """Async wrapper around :func:`scan_tokens` using aiohttp."""
    if method == "pools":
        from . import dex_scanner, scanner_common
        return await asyncio.to_thread(dex_scanner.scan_new_pools, scanner_common.SOLANA_RPC_URL)

    from .async_scanner import scan_tokens_async as _scan

    return await _scan(offline=offline)

