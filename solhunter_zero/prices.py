import os
import logging
import aiohttp
import asyncio

from typing import Iterable, Dict

from .lru import TTLCache
from .http import get_session

logger = logging.getLogger(__name__)

PRICE_API_BASE_URL = os.getenv("PRICE_API_URL", "https://price.jup.ag")
PRICE_API_PATH = "/v4/price"

# module level price cache
PRICE_CACHE_TTL = float(os.getenv("PRICE_CACHE_TTL", "30") or 30)
PRICE_CACHE = TTLCache(maxsize=256, ttl=PRICE_CACHE_TTL)


def _tokens_key(tokens: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(tokens)))


def get_cached_price(token: str) -> float | None:
    """Return cached price for ``token`` if available."""
    return PRICE_CACHE.get(token)


def update_price_cache(token: str, price: float) -> None:
    """Store ``price`` in the module cache."""
    if isinstance(price, (int, float)):
        PRICE_CACHE.set(token, float(price))




async def _fetch_prices(token_list: Iterable[str]) -> Dict[str, float]:
    ids = ",".join(token_list)
    url = f"{PRICE_API_BASE_URL}{PRICE_API_PATH}?ids={ids}"

    session = await get_session()
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

    result: Dict[str, float] = {}
    missing: list[str] = []
    for tok in token_list:
        val = get_cached_price(tok)
        if val is not None:
            result[tok] = val
        else:
            missing.append(tok)

    if missing:
        fetched = await _fetch_prices(missing)
        for t, v in fetched.items():
            update_price_cache(t, v)
            result[t] = v

    return result
