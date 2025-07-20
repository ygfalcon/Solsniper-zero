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
        if token_file:
            tokens = scan_tokens_from_file(token_file)
        elif offline:
            logger.info("Offline mode enabled, returning static tokens")
            tokens = OFFLINE_TOKENS
        else:
            from .websocket_scanner import stream_new_tokens

            gen = stream_new_tokens(SOLANA_RPC_URL)
            try:
                token = asyncio.run(anext(gen))
            except StopAsyncIteration:
                tokens = []
            finally:
                try:
                    asyncio.run(gen.aclose())
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(gen.aclose())
                    loop.close()
            if "token" in locals() and token:
                tokens = [token]
            else:
                tokens = []
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
        tokens = list(dict.fromkeys(tokens + extra))
    return tokens




async def scan_tokens_async(
    *, offline: bool = False, token_file: str | None = None, method: str = "websocket"
) -> List[str]:

    """Async wrapper around :func:`scan_tokens` using aiohttp."""

    if method == "websocket":
        if token_file:
            tokens = scan_tokens_from_file(token_file)
        elif offline:
            logger.info("Offline mode enabled, returning static tokens")
            tokens = OFFLINE_TOKENS
        else:
            from .websocket_scanner import stream_new_tokens

            gen = stream_new_tokens(SOLANA_RPC_URL)
            try:
                token = await anext(gen)
            except StopAsyncIteration:
                tokens = []
            finally:
                await gen.aclose()
            if "token" in locals() and token:
                tokens = [token]
            else:
                tokens = []
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
        tokens = list(dict.fromkeys(tokens + extra))
    return tokens

