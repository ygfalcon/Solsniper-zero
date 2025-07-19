from __future__ import annotations
import asyncio
import os
import logging
from typing import List, Dict

import aiohttp

from .scanner_onchain import scan_tokens_onchain

logger = logging.getLogger(__name__)

BIRDEYE_API = "https://public-api.birdeye.so/defi/tokenlist"
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL")
HEADERS: Dict[str, str] = {}
if BIRDEYE_API_KEY:
    HEADERS["X-API-KEY"] = BIRDEYE_API_KEY
else:
    logger.warning(
        "BIRDEYE_API_KEY not set. Falling back to on-chain scanning by default"
    )

OFFLINE_TOKENS = ["offlinebonk1", "offlinebonk2"]


async def scan_tokens_async(*, offline: bool = False) -> List[str]:
    """Async variant of :func:`scanner.scan_tokens`."""
    if offline:
        logger.info("Offline mode enabled, returning static tokens")
        return OFFLINE_TOKENS

    if not BIRDEYE_API_KEY:
        logger.info("No BirdEye API key set, scanning on-chain")
        return await asyncio.to_thread(scan_tokens_onchain, SOLANA_RPC_URL)

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
                    tokens = [
                        t["address"]
                        for t in data.get("data", [])
                        if t["address"].lower().endswith("bonk")
                    ]
                    logger.info("Found %d candidate tokens", len(tokens))
                    backoff = 1
                    return tokens
        except aiohttp.ClientError as exc:
            logger.error("Scan failed: %s", exc)
            return []
