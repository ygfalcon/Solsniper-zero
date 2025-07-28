import os
import logging
import aiohttp
import asyncio
import time

from typing import Iterable, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

PRICE_API_BASE_URL = os.getenv("PRICE_API_URL", "https://price.jup.ag")
PRICE_API_PATH = "/v4/price"

# module level session and cache
_session: Optional[aiohttp.ClientSession] = None
_cache: Dict[Tuple[str, ...], Tuple[float, Dict[str, float]]] = {}
CACHE_TTL = 30  # seconds


def _tokens_key(tokens: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(tokens)))


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or getattr(_session, "closed", False):
        _session = aiohttp.ClientSession()
    return _session


async def _fetch_prices(token_list: Iterable[str]) -> Dict[str, float]:
    ids = ",".join(token_list)
    url = f"{PRICE_API_BASE_URL}{PRICE_API_PATH}?ids={ids}"

    session = await _get_session()
    try:
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            data = (await resp.json()).get("data", {})
    except aiohttp.ClientError as exc:
        logger.warning("Failed to fetch token prices: %s", exc)
        return {}

    prices: Dict[str, float] = {}
    for token, info in data.items():
        price = info.get("price")
        if isinstance(price, (int, float)):
            prices[token] = float(price)
    return prices


def fetch_token_prices(tokens: Iterable[str]) -> Dict[str, float]:
    """Retrieve USD prices for multiple tokens from the configured API."""
    return asyncio.run(fetch_token_prices_async(tokens))


async def fetch_token_prices_async(tokens: Iterable[str]) -> Dict[str, float]:
    """Asynchronously retrieve USD prices for multiple tokens."""
    token_list = _tokens_key(tokens)
    if not token_list:
        return {}

    now = time.time()
    cached = _cache.get(token_list)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1]

    prices = await _fetch_prices(token_list)
    _cache[token_list] = (now, prices)
    return prices
