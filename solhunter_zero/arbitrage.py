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

from .exchange import (
    place_order_async,
    ORCA_DEX_URL,
    RAYDIUM_DEX_URL,
    DEX_BASE_URL,
    VENUE_URLS,
    SWAP_PATH,
)
from . import order_book_ws
from .depth_client import stream_depth, prepare_signed_tx
from .execution import EventExecutor
from .flash_loans import borrow_flash, repay_flash
from solders.instruction import Instruction
from solders.pubkey import Pubkey
from solders.keypair import Keypair

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
USE_SERVICE_EXEC = os.getenv("USE_SERVICE_EXEC", "0").lower() in {"1", "true", "yes"}


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

# Flash loan configuration
USE_FLASH_LOANS = os.getenv("USE_FLASH_LOANS", "0").lower() in {"1", "true", "yes"}
MAX_FLASH_AMOUNT = float(os.getenv("MAX_FLASH_AMOUNT", "0") or 0)


def refresh_costs() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return updated fee, gas and latency mappings from the environment."""

    fees = {str(k): float(v) for k, v in _parse_mapping_env("DEX_FEES").items()}
    gas = {str(k): float(v) for k, v in _parse_mapping_env("DEX_GAS").items()}
    latency = {str(k): float(v) for k, v in _parse_mapping_env("DEX_LATENCY").items()}
    return fees, gas, latency


async def _prepare_service_tx(
    token: str,
    side: str,
    amount: float,
    price: float,
    base_url: str,
) -> str | None:
    payload = {
        "token": token,
        "side": side,
        "amount": amount,
        "price": price,
        "cluster": "mainnet-beta",
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{base_url}{SWAP_PATH}", json=payload, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientError:
            return None

    tx_b64 = data.get("swapTransaction")
    if not tx_b64:
        return None
    return await prepare_signed_tx(tx_b64)


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
    use_flash_loans: bool | None = None,
    max_flash_amount: float | None = None,
) -> tuple[list[str], float]:
    """Return path with maximum profit and the expected profit."""

    fees = fees or {}
    gas = gas or {}
    latency = latency or {}
    if use_flash_loans is None:
        use_flash_loans = USE_FLASH_LOANS
    if max_flash_amount is None:
        max_flash_amount = MAX_FLASH_AMOUNT
    trade_amount = min(max_flash_amount or amount, amount) if use_flash_loans else amount
    venues = list(prices.keys())
    best: list[str] | None = None
    best_profit = float("-inf")

    def step_cost(a: str, b: str) -> float:
        return (
            prices[a] * trade_amount * fees.get(a, 0.0)
            + prices[b] * trade_amount * fees.get(b, 0.0)
            + gas.get(a, 0.0)
            + gas.get(b, 0.0)
            + latency.get(a, 0.0)
            + latency.get(b, 0.0)
        )

    for length in range(2, len(venues) + 1):
        for path in permutations(venues, length):
            profit = 0.0
            for i in range(len(path) - 1):
                a = path[i]
                b = path[i + 1]
                profit += (prices[b] - prices[a]) * trade_amount - step_cost(a, b)
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
    executor: "EventExecutor | None" = None,
    use_service: bool = False,
    use_flash_loans: bool | None = None,
    max_flash_amount: float | None = None,
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
    use_flash_loans:
        Borrow funds via a flash-loan program before executing the swap chain.
    max_flash_amount:
        Maximum amount to borrow when flash loans are enabled.

    Returns
    -------
    Optional[Tuple[int, int]]
        Indices of the feeds used for buy and sell orders when an opportunity is
        executed. ``None`` when no profitable opportunity is found.
    """

    if use_flash_loans is None:
        use_flash_loans = USE_FLASH_LOANS
    if max_flash_amount is None:
        max_flash_amount = MAX_FLASH_AMOUNT

    if fees is None or gas is None or latencies is None:
        env_fees, env_gas, env_lat = refresh_costs()
        if fees is None:
            fees = env_fees
        else:
            for k, v in env_fees.items():
                fees.setdefault(k, v)
        if gas is None:
            gas = env_gas
        else:
            for k, v in env_gas.items():
                gas.setdefault(k, v)
        if latencies is None:
            latencies = env_lat
        else:
            for k, v in env_lat.items():
                latencies.setdefault(k, v)
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
                use_flash_loans=use_flash_loans,
                max_flash_amount=max_flash_amount,
            )
            if not path:
                return None
            buy_name, sell_name = path[0], path[-1]
            buy_index = names.index(buy_name)
            sell_index = names.index(sell_name)
            trade_base = min(max_flash_amount or amount, amount) if use_flash_loans else amount
            diff_base = price_map[buy_name] * trade_base
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
            trade_amount = amount
            if use_flash_loans:
                flash_amt = max_flash_amount or amount
                trade_amount = min(flash_amt, amount)
                swap_ix = [Instruction(Pubkey.default(), b"swap", []) for _ in range(len(path) - 1)]
                sig = await borrow_flash(
                    trade_amount,
                    token,
                    swap_ix,
                    payer=keypair or Keypair(),
                )
                tasks = []
            else:
                sig = None
                tasks = []
                for i in range(len(path) - 1):
                    buy_v = path[i]
                    sell_v = path[i + 1]
                    if executor and use_service:
                        base_buy = VENUE_URLS.get(buy_v, buy_v)
                        base_sell = VENUE_URLS.get(sell_v, sell_v)
                        tx1 = await _prepare_service_tx(
                            token,
                            "buy",
                            trade_amount,
                            price_map[buy_v],
                            base_buy,
                        )
                        if tx1:
                            await executor.enqueue(tx1)
                        tx2 = await _prepare_service_tx(
                            token,
                            "sell",
                            trade_amount,
                            price_map[sell_v],
                            base_sell,
                        )
                        if tx2:
                            await executor.enqueue(tx2)
                    else:
                        tasks.append(
                            place_order_async(
                                token,
                                "buy",
                                trade_amount,
                                price_map[buy_v],
                                testnet=testnet,
                                dry_run=dry_run,
                                keypair=keypair,
                            )
                        )
                        tasks.append(
                            place_order_async(
                                token,
                                "sell",
                                trade_amount,
                                price_map[sell_v],
                                testnet=testnet,
                                dry_run=dry_run,
                                keypair=keypair,
                            )
                        )
                if tasks:
                    await asyncio.gather(*tasks)
            if sig:
                await repay_flash(sig)
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
        use_flash_loans=use_flash_loans,
        max_flash_amount=max_flash_amount,
    )
    if not path:
        return None
    buy_name, sell_name = path[0], path[-1]
    trade_base = min(max_flash_amount or amount, amount) if use_flash_loans else amount
    diff_base = price_map[buy_name] * trade_base
    if diff_base <= 0:
        return None
    diff = profit / diff_base
    if diff < threshold:
        logger.info("No arbitrage opportunity: diff %.4f below threshold", diff)
        return None

    logger.info(
        "Arbitrage detected on %s via %s: profit %.6f", token, "->".join(path), profit
    )

    trade_amount = amount
    sig = None
    tasks = []
    if use_flash_loans:
        flash_amt = max_flash_amount or amount
        trade_amount = min(flash_amt, amount)
        swap_ix = [Instruction(Pubkey.default(), b"swap", []) for _ in range(len(path) - 1)]
        sig = await borrow_flash(
            trade_amount,
            token,
            swap_ix,
            payer=keypair or Keypair(),
        )
    else:
        for i in range(len(path) - 1):
            buy_v = path[i]
            sell_v = path[i + 1]
            if executor and use_service:
                base_buy = VENUE_URLS.get(buy_v, buy_v)
                base_sell = VENUE_URLS.get(sell_v, sell_v)
                tx1 = await _prepare_service_tx(
                    token,
                    "buy",
                    trade_amount,
                    price_map[buy_v],
                    base_buy,
                )
                if tx1:
                    await executor.enqueue(tx1)
                tx2 = await _prepare_service_tx(
                    token,
                    "sell",
                    trade_amount,
                    price_map[sell_v],
                    base_sell,
                )
                if tx2:
                    await executor.enqueue(tx2)
            else:
                tasks.append(
                    place_order_async(
                        token,
                        "buy",
                        trade_amount,
                        price_map[buy_v],
                        testnet=testnet,
                        dry_run=dry_run,
                        keypair=keypair,
                    )
                )
                tasks.append(
                    place_order_async(
                        token,
                        "sell",
                        trade_amount,
                        price_map[sell_v],
                        testnet=testnet,
                        dry_run=dry_run,
                        keypair=keypair,
                    )
                )
        if tasks:
            await asyncio.gather(*tasks)
    if sig:
        await repay_flash(sig)

    buy_index = names.index(buy_name)
    sell_index = names.index(sell_name)
    return buy_index, sell_index


async def detect_and_execute_arbitrage(
    tokens: str | Sequence[str],
    feeds: Sequence[PriceFeed] | None = None,
    streams: Sequence[AsyncGenerator[float, None]] | None = None,
    *,
    fees: Mapping[str, float] | None = None,
    executor: EventExecutor | None = None,
    use_service: bool | None = None,
    use_flash_loans: bool | None = None,
    max_flash_amount: float | None = None,
    **kwargs,
) -> Optional[Tuple[int, int]] | list[Optional[Tuple[int, int]]]:
    """Run arbitrage detection for one or multiple tokens.

    When ``tokens`` is a sequence, price checks for each token are executed
    concurrently using :func:`asyncio.gather`.
    """

    user_gas = kwargs.pop("gas", None)
    user_lat = kwargs.pop("latencies", None)

    if use_flash_loans is None:
        use_flash_loans = USE_FLASH_LOANS
    if max_flash_amount is None:
        max_flash_amount = MAX_FLASH_AMOUNT

    if use_service is None:
        use_service = USE_SERVICE_EXEC

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
                    executor=executor,
                    use_service=use_service,
                    use_flash_loans=use_flash_loans,
                    max_flash_amount=max_flash_amount,
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
                executor=executor,
                use_service=use_service,
                use_flash_loans=use_flash_loans,
                max_flash_amount=max_flash_amount,
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
        executor=executor,
        use_service=use_service,
        use_flash_loans=use_flash_loans,
        max_flash_amount=max_flash_amount,
        **kwargs,
    )
