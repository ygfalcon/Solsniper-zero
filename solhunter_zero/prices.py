import os
import requests
import aiohttp
from typing import Iterable, Dict

PRICE_API_BASE_URL = os.getenv("PRICE_API_URL", "https://price.jup.ag")
PRICE_API_PATH = "/v4/price"


def fetch_token_prices(tokens: Iterable[str]) -> Dict[str, float]:
    """Retrieve USD prices for multiple tokens from the configured API."""
    token_list = list(tokens)
    if not token_list:
        return {}

    ids = ",".join(token_list)
    url = f"{PRICE_API_BASE_URL}{PRICE_API_PATH}?ids={ids}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", {})
    prices = {}
    for token, info in data.items():
        price = info.get("price")
        if isinstance(price, (int, float)):
            prices[token] = float(price)
    return prices


async def fetch_token_prices_async(tokens: Iterable[str]) -> Dict[str, float]:
    """Asynchronously retrieve USD prices for multiple tokens."""
    token_list = list(tokens)
    if not token_list:
        return {}

    ids = ",".join(token_list)
    url = f"{PRICE_API_BASE_URL}{PRICE_API_PATH}?ids={ids}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            data = (await resp.json()).get("data", {})

    prices: Dict[str, float] = {}
    for token, info in data.items():
        price = info.get("price")
        if isinstance(price, (int, float)):
            prices[token] = float(price)
    return prices
