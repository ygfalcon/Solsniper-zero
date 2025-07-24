import os
import logging
import asyncio
from typing import Dict, List, Optional, Iterable

import aiohttp

from pathlib import Path


from .scanner_onchain import scan_tokens_onchain

logger = logging.getLogger(__name__)

# Endpoint providing trending tokens across multiple DEXes
JUPITER_TRENDS_API = os.getenv(
    "JUPITER_TRENDS_API", "https://stats.jup.ag/trending"
)

BIRDEYE_API = "https://public-api.birdeye.so/defi/tokenlist"
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL")

# Additional discovery endpoints
RAYDIUM_LISTINGS_API = os.getenv(
    "RAYDIUM_LISTINGS_API", "https://api.raydium.io/new-listings"
)
ORCA_LISTINGS_API = os.getenv(
    "ORCA_LISTINGS_API", "https://api.orca.so/new-listings"
)
PHOENIX_LISTINGS_API = os.getenv(
    "PHOENIX_LISTINGS_API", "https://api.phoenix.trade/new-listings"
)
METEORA_LISTINGS_API = os.getenv(
    "METEORA_LISTINGS_API", "https://api.meteora.ag/new-listings"
)
JUPITER_WS_URL = os.getenv("JUPITER_WS_URL", "wss://stats.jup.ag/ws")
PHOENIX_WS_URL = os.getenv("PHOENIX_WS_URL", "")
METEORA_WS_URL = os.getenv("METEORA_WS_URL", "")
DEX_LISTING_WS_URL = os.getenv("DEX_LISTING_WS_URL", "")

# Filtering configuration
TOKEN_SUFFIX = os.getenv("TOKEN_SUFFIX", "bonk")
TOKEN_KEYWORDS: List[str] = [
    k.strip().lower()
    for k in os.getenv("TOKEN_KEYWORDS", "").split(",")
    if k.strip()
]
VOLUME_THRESHOLD = float(os.getenv("VOLUME_THRESHOLD", "0") or 0)

HEADERS: Dict[str, str] = {}
if BIRDEYE_API_KEY:
    HEADERS["X-API-KEY"] = BIRDEYE_API_KEY
else:
    logger.warning(
        "BIRDEYE_API_KEY not set. Falling back to on-chain scanning by default"
    )

OFFLINE_TOKENS = ["offlinebonk1", "offlinebonk2"]


def token_matches(
    address: str,
    name: str | None = None,
    volume: float | None = None,
    *,
    suffix: str | None = None,
    keywords: Iterable[str] | None = None,
    volume_threshold: float | None = None,
) -> bool:
    """Return ``True`` if token passes configured filters."""

    if volume_threshold is None:
        volume_threshold = VOLUME_THRESHOLD
    if volume is not None and volume_threshold and float(volume) < volume_threshold:
        return False

    if keywords is None:
        keywords = TOKEN_KEYWORDS
    if suffix is None:
        suffix = TOKEN_SUFFIX

    target = (name or address).lower()

    if keywords:
        for kw in keywords:
            if kw and kw.lower() in target:
                return True
        return False

    if suffix:
        return target.endswith(suffix.lower())

    return True


def load_tokens_from_file(path: str) -> List[str]:
    """Return token addresses listed one per line in *path*."""
    tokens: List[str] = []
    for line in Path(path).read_text().splitlines():
        tok = line.strip()
        if tok and not tok.startswith("#"):
            tokens.append(tok)
    logger.info("Loaded %d tokens from %s", len(tokens), path)
    return tokens


def parse_birdeye_tokens(data: dict) -> List[str]:
    tokens: List[str] = []
    for t in data.get("data", []):
        addr = t.get("address")
        name = t.get("name") or t.get("symbol")
        volume = t.get("volume") or t.get("volume_24h") or t.get("vol24h")
        if addr and token_matches(addr, name, volume):
            tokens.append(addr)

    logger.info("Found %d candidate tokens", len(tokens))
    return tokens


def parse_trending_tokens(data: dict) -> List[str]:
    """Extract token addresses from the Jupiter trends API response."""
    token_list = []
    if isinstance(data, list):
        token_list = data
    else:
        for key in ("trending", "data", "tokens"):
            if isinstance(data.get(key), list):
                token_list = data[key]
                break

    tokens: List[str] = []
    for entry in token_list:
        addr = entry.get("address") or entry.get("id") or entry.get("mint")
        name = entry.get("name") or entry.get("symbol")
        vol = entry.get("volume") or entry.get("volume_24h") or entry.get("vol24h")
        if addr and token_matches(addr, name, vol):
            tokens.append(addr)
    logger.info("Found %d trending tokens", len(tokens))
    return tokens


def fetch_trending_tokens() -> List[str]:
    """Fetch trending token addresses from Jupiter using ``aiohttp``."""
    return asyncio.run(fetch_trending_tokens_async())


async def fetch_trending_tokens_async() -> List[str]:
    """Asynchronously fetch trending token addresses from Jupiter."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(JUPITER_TRENDS_API, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientError as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch trending tokens: %s", exc)
            return []

    return parse_trending_tokens(data)


def parse_listing_tokens(data: dict) -> List[str]:
    """Return token addresses from DEX listing APIs respecting filters."""
    tokens: List[str] = []
    for entry in data.get("data", []):
        addr = entry.get("address") or entry.get("mint")
        name = entry.get("name") or entry.get("symbol")
        vol = entry.get("volume") or entry.get("volume_24h") or entry.get("vol24h")
        if addr and token_matches(addr, name, vol):
            tokens.append(addr)
    logger.info("Found %d tokens from listings", len(tokens))
    return tokens


def fetch_raydium_listings() -> List[str]:
    """Return token listings from the Raydium API using ``aiohttp``."""
    return asyncio.run(fetch_raydium_listings_async())


async def fetch_raydium_listings_async() -> List[str]:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(RAYDIUM_LISTINGS_API, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientError as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch Raydium listings: %s", exc)
            return []
    return parse_listing_tokens(data)


def fetch_orca_listings() -> List[str]:
    """Return token listings from the Orca API using ``aiohttp``."""
    return asyncio.run(fetch_orca_listings_async())


async def fetch_orca_listings_async() -> List[str]:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(ORCA_LISTINGS_API, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientError as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch Orca listings: %s", exc)
            return []
    return parse_listing_tokens(data)


def fetch_phoenix_listings() -> List[str]:
    """Return token listings from the Phoenix API using ``aiohttp``."""
    return asyncio.run(fetch_phoenix_listings_async())


async def fetch_phoenix_listings_async() -> List[str]:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(PHOENIX_LISTINGS_API, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientError as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch Phoenix listings: %s", exc)
            return []
    return parse_listing_tokens(data)


def fetch_meteora_listings() -> List[str]:
    """Return token listings from the Meteora API using ``aiohttp``."""
    return asyncio.run(fetch_meteora_listings_async())


async def fetch_meteora_listings_async() -> List[str]:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(METEORA_LISTINGS_API, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientError as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch Meteora listings: %s", exc)
            return []
    return parse_listing_tokens(data)


def offline_or_onchain(
    offline: bool,
    token_file: str | None = None,
    *,
    method: str | None = None,
) -> Optional[List[str]]:
    if token_file:
        return load_tokens_from_file(token_file)
    if offline:
        logger.info("Offline mode enabled, returning static tokens")
        return OFFLINE_TOKENS

    if not BIRDEYE_API_KEY:
        logger.info("No BirdEye API key set, scanning on-chain")
        if method == "pools":
            return scan_tokens_from_pools()
        if method == "file":
            return scan_tokens_from_file()
        return scan_tokens_onchain(SOLANA_RPC_URL)

    return None



async def offline_or_onchain_async(

    offline: bool,
    token_file: str | None = None,
    *,
    method: str = "rest",
) -> Optional[List[str]]:
    """Return tokens from file or on-chain when BirdEye is unavailable."""

    if token_file:
        return load_tokens_from_file(token_file)

    if offline:
        logger.info("Offline mode enabled, returning static tokens")
        return OFFLINE_TOKENS

    if method == "onchain":
        return await asyncio.to_thread(scan_tokens_onchain, SOLANA_RPC_URL)
    if method == "pools":
        return await asyncio.to_thread(scan_tokens_from_pools)
    if method == "file":
        return await asyncio.to_thread(scan_tokens_from_file)
    if method == "mempool":
        from .mempool_scanner import stream_mempool_tokens

        gen = stream_mempool_tokens(SOLANA_RPC_URL)
        try:
            token = await anext(gen)
        except StopAsyncIteration:
            return []
        finally:
            await gen.aclose()

        return [token]

    if not BIRDEYE_API_KEY:
        logger.info("No BirdEye API key set, scanning on-chain")
        if method == "websocket":
            from .websocket_scanner import stream_new_tokens

            gen = stream_new_tokens(SOLANA_RPC_URL)
            try:
                token = await anext(gen)
            except StopAsyncIteration:
                return []
            finally:
                await gen.aclose()

            return [token]

        if method == "pools":
            return await asyncio.to_thread(scan_tokens_from_pools)
        if method == "file":
            return await asyncio.to_thread(scan_tokens_from_file)
        return await asyncio.to_thread(scan_tokens_onchain, SOLANA_RPC_URL)

    return None


def scan_tokens_from_pools() -> List[str]:

    """Discover tokens from recently created liquidity pools."""

    logger.info("Scanning pools for tokens")
    from . import dex_scanner

    return dex_scanner.scan_new_pools(SOLANA_RPC_URL)


def scan_tokens_from_file(path: str = "tokens.txt") -> List[str]:
    """Load token list from a file if it exists."""
    if not os.path.isfile(path):
        logger.warning("Token file %s not found", path)
        return []
    return load_tokens_from_file(path)
