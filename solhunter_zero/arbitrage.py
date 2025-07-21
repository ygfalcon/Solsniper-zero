import asyncio
import logging
import os
import json
from typing import (
    Callable,
    Awaitable,
    Sequence,
    Tuple,
    Optional,
    AsyncGenerator,
    Mapping,
)

import aiohttp
from .scanner_common import JUPITER_WS_URL

from .exchange import place_order_async
from . import order_book_ws

logger = logging.getLogger(__name__)

DEPTH_SERVICE_SOCKET = os.getenv("DEPTH_SERVICE_SOCKET", "/tmp/depth_service.sock")

PriceFeed = Callable[[str], Awaitable[float]]


# Default API endpoints for direct price queries
ORCA_API_URL = os.getenv("ORCA_API_URL", "https://api.orca.so")
RAYDIUM_API_URL = os.getenv("RAYDIUM_API_URL", "https://api.raydium.io")
ORCA_WS_URL = os.getenv("ORCA_WS_URL", "")
RAYDIUM_WS_URL = os.getenv("RAYDIUM_WS_URL", "")
DEX_PRIORITIES = [
    n.strip()
    for n in os.getenv("DEX_PRIORITIES", "service,orca,raydium,jupiter")
    .replace(";", ",")
    .split(",")
    if n.strip()
]

def _parse_fee_env() -> dict:
    val = os.getenv("DEX_FEES")
    if not val:
        return {}
    try:
        data = json.loads(val)
    except Exception:
        try:
            import ast
            data = ast.literal_eval(val)
        except Exception:
            return {}
    try:
        return {str(k): float(v) for k, v in data.items()}
    except Exception:
        return {}

DEX_FEES = _parse_fee_env()


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


async def stream_orca_prices(token: str, url: str = ORCA_WS_URL) -> AsyncGenerator[float, None]:
    """Yield live prices for ``token`` from the Orca websocket feed."""

    if not url:
        return

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            try:
                await ws.send_str(json.dumps({"token": token}))
            except Exception:  # pragma: no cover - send failures
                pass
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except Exception:  # pragma: no cover - invalid message
                    continue
                price = data.get("price")
                if isinstance(price, (int, float)):
                    yield float(price)


async def stream_raydium_prices(token: str, url: str = RAYDIUM_WS_URL) -> AsyncGenerator[float, None]:
    """Yield live prices for ``token`` from the Raydium websocket feed."""

    if not url:
        return

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            try:
                await ws.send_str(json.dumps({"token": token}))
            except Exception:  # pragma: no cover - send failures
                pass
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except Exception:  # pragma: no cover - invalid message
                    continue
                price = data.get("price")
                if isinstance(price, (int, float)):
                    yield float(price)


async def stream_jupiter_prices(token: str, url: str = JUPITER_WS_URL) -> AsyncGenerator[float, None]:
    """Yield live prices for ``token`` from the Jupiter websocket feed."""

    if not url:
        return

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            try:
                await ws.send_str(json.dumps({"token": token}))
            except Exception:  # pragma: no cover - send failures
                pass
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except Exception:  # pragma: no cover - invalid message
                    continue
                price = data.get("price")
                if isinstance(price, (int, float)):
                    yield float(price)


async def _detect_for_token(
    token: str,
    feeds: Sequence[PriceFeed] | None = None,
    streams: Sequence[AsyncGenerator[float, None]] | None = None,
    *,
    threshold: float = 0.0,
    amount: float = 1.0,
    testnet: bool = False,
    dry_run: bool = False,
    keypair=None,
    max_updates: int | None = None,
    fees: Mapping[str, float] | None = None,
    stream_names: Sequence[str] | None = None,
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

    if streams:
        prices: list[Optional[float]] = [None] * len(streams)
        result: Optional[Tuple[int, int]] = None

        async def maybe_execute() -> Optional[Tuple[int, int]]:
            if any(p is None for p in prices):
                return None
            min_price = min(p for p in prices if p is not None)
            max_price = max(p for p in prices if p is not None)
            buy_index = prices.index(min_price)
            sell_index = prices.index(max_price)
            if sell_index == buy_index or min_price <= 0:
                return None
            buy_name = (
                stream_names[buy_index]
                if stream_names and buy_index < len(stream_names)
                else str(buy_index)
            )
            sell_name = (
                stream_names[sell_index]
                if stream_names and sell_index < len(stream_names)
                else str(sell_index)
            )
            diff = (max_price - min_price) / min_price
            if fees:
                diff -= fees.get(buy_name, 0.0) + fees.get(sell_name, 0.0)
            if diff < threshold:
                return None
            logger.info(
                "Arbitrage detected on %s: buy at %.6f sell at %.6f",
                token,
                min_price,
                max_price,
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

        async def consume(idx: int, gen: AsyncGenerator[float, None]):
            nonlocal result
            count = 0
            async for price in gen:
                if result is not None:
                    break
                prices[idx] = price
                res = await maybe_execute()
                if res:
                    result = res
                    break
                count += 1
                if max_updates is not None and count >= max_updates:
                    break

        tasks = [asyncio.create_task(consume(i, g)) for i, g in enumerate(streams)]
        await asyncio.gather(*tasks, return_exceptions=True)
        return result

    if not feeds:
        feeds = [fetch_orca_price_async, fetch_raydium_price_async]

    prices = await asyncio.gather(*(feed(token) for feed in feeds))
    if not prices:
        return None

    min_price = min(prices)
    max_price = max(prices)
    buy_index = prices.index(min_price)
    sell_index = prices.index(max_price)
    names = [getattr(f, "__name__", str(i)) for i, f in enumerate(feeds)] if feeds else []
    buy_name = names[buy_index] if names else str(buy_index)
    sell_name = names[sell_index] if names else str(sell_index)

    if sell_index == buy_index:
        return None

    if min_price <= 0:
        return None

    diff = (max_price - min_price) / min_price
    if fees:
        diff -= fees.get(buy_name, 0.0) + fees.get(sell_name, 0.0)
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


async def detect_and_execute_arbitrage(
    tokens: str | Sequence[str],
    feeds: Sequence[PriceFeed] | None = None,
    streams: Sequence[AsyncGenerator[float, None]] | None = None,
    *,
    fees: Mapping[str, float] | None = None,
    **kwargs,
) -> Optional[Tuple[int, int]] | list[Optional[Tuple[int, int]]]:
    """Run arbitrage detection for one or multiple tokens.

    When ``tokens`` is a sequence, price checks for each token are executed
    concurrently using :func:`asyncio.gather`.
    """

    def _streams_for(token: str):
        if streams is not None:
            return streams, None

        available = {
            "orca": (ORCA_WS_URL, stream_orca_prices),
            "raydium": (RAYDIUM_WS_URL, stream_raydium_prices),
            "jupiter": (JUPITER_WS_URL, stream_jupiter_prices),
            "service": (f"ipc://{DEPTH_SERVICE_SOCKET}?{token}", order_book_ws.stream_order_book),
        }
        auto: list[AsyncGenerator[float, None]] = []
        names: list[str] = []
        for name in DEX_PRIORITIES:
            url, fn = available.get(name, ("", None))
            if url and fn is not None:
                auto.append(fn(token, url=url))
                names.append(name)
        if not auto:
            return None, None
        return auto, names

    if isinstance(tokens, Sequence) and not isinstance(tokens, str):
        tasks = []
        for t in tokens:
            s, names = _streams_for(t)
            tasks.append(
                _detect_for_token(
                    t,
                    feeds=feeds,
                    streams=s,
                    stream_names=names,
                    fees=fees or DEX_FEES,
                    **kwargs,
                )
            )
        
        return await asyncio.gather(*tasks)

    s, names = _streams_for(tokens)
    return await _detect_for_token(
        tokens,
        feeds=feeds,
        streams=s,
        stream_names=names,
        fees=fees or DEX_FEES,
        **kwargs,
    )
