import asyncio
import logging
import os
from typing import Callable, Awaitable, Sequence, Tuple, Optional

import aiohttp

from .exchange import place_order_async

logger = logging.getLogger(__name__)

PriceFeed = Callable[[str], Awaitable[float]]


# Default API endpoints for direct price queries
ORCA_API_URL = os.getenv("ORCA_API_URL", "https://api.orca.so")
RAYDIUM_API_URL = os.getenv("RAYDIUM_API_URL", "https://api.raydium.io")


async def fetch_orca_price_async(token: str) -> float:
    """Return the current price for ``token`` from the Orca API."""

    url = f"{ORCA_API_URL}/price?token={token}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
                price = data.get("price")
                return float(price) if isinstance(price, (int, float)) else 0.0
        except aiohttp.ClientError as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch price from Orca: %s", exc)
            return 0.0


async def fetch_raydium_price_async(token: str) -> float:
    """Return the current price for ``token`` from the Raydium API."""

    url = f"{RAYDIUM_API_URL}/price?token={token}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
                price = data.get("price")
                return float(price) if isinstance(price, (int, float)) else 0.0
        except aiohttp.ClientError as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch price from Raydium: %s", exc)
            return 0.0


async def detect_and_execute_arbitrage(
    token: str,
    feeds: Sequence[PriceFeed] | None = None,
    *,
    threshold: float = 0.0,
    amount: float = 1.0,
    testnet: bool = False,
    dry_run: bool = False,
    keypair=None,
) -> Optional[Tuple[int, int]]:
    """Check for price discrepancies and place arbitrage orders.

    Parameters
    ----------
    token:
        Token symbol or address to trade.
    feeds:
        Sequence of callables returning the token price from different DEXes.
    threshold:
        Minimum fractional price difference required to trigger arbitrage.
    amount:
        Trade size to use for buy/sell orders.

    Returns
    -------
    Optional[Tuple[int, int]]
        Indices of the feeds used for buy and sell orders when an opportunity is
        executed. ``None`` when no profitable opportunity is found.
    """

    if not feeds:
        feeds = [fetch_orca_price_async, fetch_raydium_price_async]

    prices = await asyncio.gather(*(feed(token) for feed in feeds))
    if not prices:
        return None

    min_price = min(prices)
    max_price = max(prices)
    buy_index = prices.index(min_price)
    sell_index = prices.index(max_price)

    if sell_index == buy_index:
        return None

    if min_price <= 0:
        return None

    diff = (max_price - min_price) / min_price
    if diff < threshold:
        logger.info("No arbitrage opportunity: diff %.4f below threshold", diff)
        return None

    logger.info(
        "Arbitrage detected on %s: buy at %.6f sell at %.6f", token, min_price, max_price
    )

    await place_order_async(
        token,
        "buy",
        amount,
        min_price,
        testnet=testnet,
        dry_run=dry_run,
        keypair=keypair,
    )
    await place_order_async(
        token,
        "sell",
        amount,
        max_price,
        testnet=testnet,
        dry_run=dry_run,
        keypair=keypair,
    )

    return buy_index, sell_index
