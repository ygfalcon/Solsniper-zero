import asyncio
import logging
import os
import json
from itertools import permutations
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
from .depth_client import stream_depth

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

USE_DEPTH_STREAM = os.getenv("USE_DEPTH_STREAM", "0").lower() in {"1", "true", "yes"}


def _parse_mapping_env(env: str) -> dict:
    """Return dictionary from ``env`` or an empty mapping."""
    val = os.getenv(env)
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
    if isinstance(data, dict):
        return data
    return {}


DEX_FEES = {str(k): float(v) for k, v in _parse_mapping_env("DEX_FEES").items()}
DEX_GAS = {str(k): float(v) for k, v in _parse_mapping_env("DEX_GAS").items()}
DEX_LATENCY = {str(k): float(v) for k, v in _parse_mapping_env("DEX_LATENCY").items()}
EXTRA_API_URLS = {str(k): str(v) for k, v in _parse_mapping_env("DEX_API_URLS").items()}
EXTRA_WS_URLS = {str(k): str(v) for k, v in _parse_mapping_env("DEX_WS_URLS").items()}


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


JUPITER_API_URL = os.getenv("JUPITER_API_URL", "https://price.jup.ag/v4/price")


async def fetch_jupiter_price_async(token: str) -> float:
    """Return the current price for ``token`` from the Jupiter price API."""

    url = f"{JUPITER_API_URL}?ids={token}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
                info = data.get("data", {}).get(token)
                if info and isinstance(info, dict):
                    price = info.get("price")
                    return float(price) if isinstance(price, (int, float)) else 0.0
        except aiohttp.ClientError as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch price from Jupiter: %s", exc)
            return 0.0


def make_api_price_fetch(url: str) -> PriceFeed:
    """Return a simple price fetcher for ``url``."""

    async def _fetch(token: str) -> float:
        req = f"{url.rstrip('/')}/price?token={token}"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(req, timeout=10) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    price = data.get("price")
                    return float(price) if isinstance(price, (int, float)) else 0.0
            except aiohttp.ClientError as exc:  # pragma: no cover - network errors
                logger.warning("Failed to fetch price from %s: %s", url, exc)
                return 0.0

    return _fetch


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


def make_ws_stream(url: str) -> Callable[[str], AsyncGenerator[float, None]]:
    """Return a simple websocket price stream factory."""

    async def _stream(token: str, url: str = url) -> AsyncGenerator[float, None]:
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

    return _stream


def _best_route(
    prices: Mapping[str, float],
    amount: float,
    *,
    fees: Mapping[str, float] | None = None,
    gas: Mapping[str, float] | None = None,
    latency: Mapping[str, float] | None = None,
) -> tuple[list[str], float]:
    """Return path with maximum profit and the expected profit."""

    fees = fees or {}
    gas = gas or {}
    latency = latency or {}
    venues = list(prices.keys())
    best: list[str] | None = None
    best_profit = float("-inf")

    def step_cost(a: str, b: str) -> float:
        return (
            prices[a] * amount * fees.get(a, 0.0)
            + prices[b] * amount * fees.get(b, 0.0)
            + gas.get(a, 0.0)
            + gas.get(b, 0.0)
            + latency.get(a, 0.0)
            + latency.get(b, 0.0)
        )

    for length in range(2, min(3, len(venues)) + 1):
        for path in permutations(venues, length):
            profit = 0.0
            for i in range(len(path) - 1):
                a = path[i]
                b = path[i + 1]
                profit += (prices[b] - prices[a]) * amount - step_cost(a, b)
            if profit > best_profit:
                best_profit = profit
                best = list(path)
    return (best or []), best_profit


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
    gas: Mapping[str, float] | None = None,
    latencies: Mapping[str, float] | None = None,
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
            names = [
                stream_names[i] if stream_names and i < len(stream_names) else str(i)
                for i in range(len(prices))
            ]
            price_map = {
                n: p for n, p in zip(names, prices) if p is not None and p > 0
            }
            if len(price_map) < 2:
                return None
            path, profit = _best_route(
                price_map,
                amount,
                fees=fees,
                gas=gas,
                latency=latencies,
            )
            if not path:
                return None
            buy_name, sell_name = path[0], path[-1]
            buy_index = names.index(buy_name)
            sell_index = names.index(sell_name)
            diff_base = price_map[buy_name] * amount
            if diff_base <= 0:
                return None
            diff = profit / diff_base
            if diff < threshold:
                return None
            logger.info(
                "Arbitrage detected on %s via %s: profit %.6f",
                token,
                "->".join(path),
                profit,
            )
            tasks = []
            first = path[0]
            last = path[-1]
            tasks.append(
                place_order_async(
                    token,
                    "buy",
                    amount,
                    price_map[first],
                    testnet=testnet,
                    dry_run=dry_run,
                    keypair=keypair,
                )
            )
            tasks.append(
                place_order_async(
                    token,
                    "sell",
                    amount,
                    price_map[last],
                    testnet=testnet,
                    dry_run=dry_run,
                    keypair=keypair,
                )
            )
            await asyncio.gather(*tasks)
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
        feeds = [fetch_orca_price_async, fetch_raydium_price_async, fetch_jupiter_price_async]
        for name, url in EXTRA_API_URLS.items():
            feeds.append(make_api_price_fetch(url))
            if fees is not None and name not in fees:
                fees[name] = DEX_FEES.get(name, 0.0)
            if gas is not None and name not in gas:
                gas[name] = DEX_GAS.get(name, 0.0)
            if latencies is not None and name not in latencies:
                latencies[name] = DEX_LATENCY.get(name, 0.0)

    prices = await asyncio.gather(*(feed(token) for feed in feeds))
    if not prices:
        return None

    names = [getattr(f, "__name__", str(i)) for i, f in enumerate(feeds)] if feeds else []
    price_map = {n: p for n, p in zip(names, prices) if p > 0}
    if len(price_map) < 2:
        return None

    path, profit = _best_route(
        price_map,
        amount,
        fees=fees,
        gas=gas,
        latency=latencies,
    )
    if not path:
        return None
    buy_name, sell_name = path[0], path[-1]
    diff_base = price_map[buy_name] * amount
    if diff_base <= 0:
        return None
    diff = profit / diff_base
    if diff < threshold:
        logger.info("No arbitrage opportunity: diff %.4f below threshold", diff)
        return None

    logger.info(
        "Arbitrage detected on %s via %s: profit %.6f", token, "->".join(path), profit
    )

    tasks = [
        place_order_async(
            token,
            "buy",
            amount,
            price_map[buy_name],
            testnet=testnet,
            dry_run=dry_run,
            keypair=keypair,
        ),
        place_order_async(
            token,
            "sell",
            amount,
            price_map[sell_name],
            testnet=testnet,
            dry_run=dry_run,
            keypair=keypair,
        ),
    ]
    await asyncio.gather(*tasks)

    buy_index = names.index(buy_name)
    sell_index = names.index(sell_name)
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

    user_gas = kwargs.pop("gas", None)
    user_lat = kwargs.pop("latencies", None)

    def _streams_for(token: str):
        if streams is not None:
            return streams, None
        if feeds is not None:
            return None, None

        available = {
            "orca": (ORCA_WS_URL, stream_orca_prices),
            "raydium": (RAYDIUM_WS_URL, stream_raydium_prices),
            "jupiter": (JUPITER_WS_URL, stream_jupiter_prices),
            "service": (f"ipc://{DEPTH_SERVICE_SOCKET}?{token}", None),
        }
        for name, url in EXTRA_WS_URLS.items():
            available[name] = (url, make_ws_stream(url))
        auto: list[AsyncGenerator[float, None]] = []
        names: list[str] = []
        for name in DEX_PRIORITIES:
            url, fn = available.get(name, ("", None))
            if not url:
                continue
            if name == "service":
                auto.append(order_book_ws.stream_order_book(url))
                names.append(name)
                continue
            if fn is not None:
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
                    gas=user_gas or DEX_GAS,
                    latencies=user_lat or DEX_LATENCY,
                    **kwargs,
                )
            )
        
        return await asyncio.gather(*tasks)

    s, names = _streams_for(tokens)
    if USE_DEPTH_STREAM:
        async for _ in stream_depth(tokens, max_updates=kwargs.get("max_updates")):
            res = await _detect_for_token(
                tokens,
                feeds=feeds,
                streams=s,
                stream_names=names,
                fees=fees or DEX_FEES,
                gas=user_gas or DEX_GAS,
                latencies=user_lat or DEX_LATENCY,
                **kwargs,
            )
            if res:
                return res
        return None

    return await _detect_for_token(
        tokens,
        feeds=feeds,
        streams=s,
        stream_names=names,
        fees=fees or DEX_FEES,
        gas=user_gas or DEX_GAS,
        latencies=user_lat or DEX_LATENCY,
        **kwargs,
    )
