from __future__ import annotations
import asyncio
import logging
from typing import List

import aiohttp

from .scanner_common import (
    BIRDEYE_API,
    HEADERS,
    offline_or_onchain_async,
    parse_birdeye_tokens,
)

logger = logging.getLogger(__name__)


async def scan_tokens_async(*, offline: bool = False, token_file: str | None = None) -> List[str]:
    """Async variant of :func:`scanner.scan_tokens`."""
    tokens = await offline_or_onchain_async(offline, token_file)
    if tokens is not None:
        return tokens

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
                    tokens = parse_birdeye_tokens(data)
                    backoff = 1
                    return tokens
        except aiohttp.ClientError as exc:
            logger.error("Scan failed: %s", exc)
            return []
