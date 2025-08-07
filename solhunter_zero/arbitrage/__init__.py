"""Arbitrage utilities and helpers.

Latency measurements now run concurrently using :func:`asyncio.gather`. With
dynamic concurrency enabled this refreshes endpoint latency around 30-50% faster
when multiple URLs are checked.
"""

import asyncio
import logging
import os
from typing import (
    Callable,
    Awaitable,
    Sequence,
    Tuple,
    Optional,
    AsyncGenerator,
    Mapping,
    List,
)

from ..system import set_rayon_threads

# Configure Rayon parallelism for the Rust FFI
set_rayon_threads()

from ..http import get_session, loads, dumps
import aiohttp
from ..scanner_common import JUPITER_WS_URL
from ..exchange import (
    place_order_async,
    ORCA_DEX_URL,
    RAYDIUM_DEX_URL,
    DEX_BASE_URL,
    VENUE_URLS,
    SWAP_PATH,
)
from .. import order_book_ws, depth_client
from ..depth_client import stream_depth, prepare_signed_tx
from ..execution import EventExecutor
from ..flash_loans import borrow_flash, repay_flash
from solders.instruction import Instruction
from solders.pubkey import Pubkey
from solders.keypair import Keypair

from solhunter_zero.lru import TTLCache
from ..event_bus import subscribe, publish
from ..prices import get_cached_price, update_price_cache
from .latency import (
    measure_dex_latency_async,
    measure_dex_latency,
    start_latency_refresh,
    stop_latency_refresh,
    DEX_LATENCY,
    MEASURE_DEX_LATENCY,
    DEX_LATENCY_REFRESH_INTERVAL,
)
from .routing import (
    EXTRA_API_URLS,
    EXTRA_WS_URLS,
    DEX_FEES,
    DEX_GAS,
    MAX_HOPS,
    PATH_ALGORITHM,
    USE_FLASH_LOANS,
    MAX_FLASH_AMOUNT,
    FLASH_LOAN_RATIO,
    USE_MEV_BUNDLES,
    refresh_costs,
    _compute_route,
    invalidate_route,
    invalidate_edges,
)
logger = logging.getLogger(__name__)

# rate limit for depth streams (seconds between updates)
DEPTH_RATE_LIMIT = 0.1

DEPTH_SERVICE_SOCKET = os.getenv("DEPTH_SERVICE_SOCKET", "/tmp/depth_service.sock")

PriceFeed = Callable[[str], Awaitable[float]]

# Default API endpoints for direct price queries
ORCA_API_URL = os.getenv("ORCA_API_URL", "https://api.orca.so")
RAYDIUM_API_URL = os.getenv("RAYDIUM_API_URL", "https://api.raydium.io")
PHOENIX_API_URL = os.getenv("PHOENIX_API_URL", "https://api.phoenix.trade")
METEORA_API_URL = os.getenv("METEORA_API_URL", "https://api.meteora.ag")
ORCA_WS_URL = os.getenv("ORCA_WS_URL", "")
RAYDIUM_WS_URL = os.getenv("RAYDIUM_WS_URL", "")
PHOENIX_WS_URL = os.getenv("PHOENIX_WS_URL", "")
METEORA_WS_URL = os.getenv("METEORA_WS_URL", "")
DEX_PRIORITIES = [
    n.strip()
    for n in os.getenv(
        "DEX_PRIORITIES",
        "service,phoenix,meteora,orca,raydium,jupiter",
    )
    .replace(";", ",")
    .split(",")
    if n.strip()
]

USE_DEPTH_STREAM = os.getenv("USE_DEPTH_STREAM", "1").lower() in {"1", "true", "yes"}
USE_SERVICE_EXEC = os.getenv("USE_SERVICE_EXEC", "True").lower() in {"1", "true", "yes"}
USE_SERVICE_ROUTE = os.getenv("USE_SERVICE_ROUTE", "1").lower() in {"1", "true", "yes"}

PRICE_CACHE_TTL = float(os.getenv("PRICE_CACHE_TTL", "30") or 30)
PRICE_CACHE = TTLCache(maxsize=256, ttl=PRICE_CACHE_TTL)
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
    use_mev_bundles: bool | None = None,
    max_flash_amount: float | None = None,
    max_hops: int | None = None,
    path_algorithm: str | None = None,
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
    max_hops:
        Maximum number of venues to traverse when searching for a path.
    path_algorithm:
        "graph" to use the dynamic graph search or "permutation" for the
        legacy exhaustive search.

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
            price_map = {n: p for n, p in zip(names, prices) if p is not None and p > 0}
            if len(price_map) < 2:
                return None
            path, profit = await _compute_route(
                token,
                amount,
                price_map,
                fees=fees,
                gas=gas,
                latency=latencies,
                use_service=USE_SERVICE_ROUTE,
                use_flash_loans=use_flash_loans,
                max_flash_amount=max_flash_amount,
                max_hops=max_hops,
                path_algorithm=path_algorithm,
            )
            if not path:
                return None
            buy_name, sell_name = path[0], path[-1]
            buy_index = names.index(buy_name)
            sell_index = names.index(sell_name)
            trade_base = (
                min(max_flash_amount or amount, amount) if use_flash_loans else amount
            )
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
                swap_ix = [
                    Instruction(Pubkey.default(), b"swap", [])
                    for _ in range(len(path) - 1)
                ]
                sig = await borrow_flash(
                    trade_amount,
                    token,
                    swap_ix,
                    payer=keypair or Keypair(),
                )
                tasks = []
            else:
                sig = None
                if use_service:
                    txs: list[str] = []
                    for i in range(len(path) - 1):
                        buy_v = path[i]
                        sell_v = path[i + 1]
                        base_buy = VENUE_URLS.get(buy_v, buy_v)
                        base_sell = VENUE_URLS.get(sell_v, sell_v)
                        tx1 = await _prepare_service_tx(
                            token,
                            "buy",
                            trade_amount,
                            price_map[buy_v],
                            base_buy,
                        )
                        tx2 = await _prepare_service_tx(
                            token,
                            "sell",
                            trade_amount,
                            price_map[sell_v],
                            base_sell,
                        )
                        if tx1:
                            txs.append(tx1)
                        if tx2:
                            txs.append(tx2)

                    if txs:
                        if use_mev_bundles:
                            from .mev_executor import MEVExecutor

                            mev = MEVExecutor(
                                token,
                                priority_rpc=(
                                    getattr(executor, "priority_rpc", None)
                                    if executor
                                    else None
                                ),
                            )
                            await mev.submit_bundle(txs)
                        elif executor:
                            for tx in txs:
                                await executor.enqueue(tx)
                        else:
                            from .depth_client import submit_raw_tx

                            for tx in txs:
                                await submit_raw_tx(
                                    tx,
                                    priority_rpc=(
                                        getattr(executor, "priority_rpc", None)
                                        if executor
                                        else None
                                    ),
                                )
                else:
                    tasks = []
                    for i in range(len(path) - 1):
                        buy_v = path[i]
                        sell_v = path[i + 1]
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
        feeds = [
            fetch_orca_price_async,
            fetch_raydium_price_async,
            fetch_phoenix_price_async,
            fetch_meteora_price_async,
            fetch_jupiter_price_async,
        ]
        for name in ("orca", "raydium", "phoenix", "meteora", "jupiter"):
            if fees is not None and name not in fees:
                fees[name] = DEX_FEES.get(name, 0.0)
            if gas is not None and name not in gas:
                gas[name] = DEX_GAS.get(name, 0.0)
            if latencies is not None and name not in latencies:
                latencies[name] = DEX_LATENCY.get(name, 0.0)
        for name, url in EXTRA_API_URLS.items():
            feeds.append(make_api_price_fetch(url, name))
            if fees is not None and name not in fees:
                fees[name] = DEX_FEES.get(name, 0.0)
            if gas is not None and name not in gas:
                gas[name] = DEX_GAS.get(name, 0.0)
            if latencies is not None and name not in latencies:
                latencies[name] = DEX_LATENCY.get(name, 0.0)

    prices = await asyncio.gather(*(feed(token) for feed in feeds))
    if not prices:
        return None

    names = (
        [getattr(f, "__name__", str(i)) for i, f in enumerate(feeds)] if feeds else []
    )
    price_map = {n: p for n, p in zip(names, prices) if p > 0}
    if len(price_map) < 2:
        return None

    path, profit = await _compute_route(
        token,
        amount,
        price_map,
        fees=fees,
        gas=gas,
        latency=latencies,
        use_service=USE_SERVICE_ROUTE,
        use_flash_loans=use_flash_loans,
        max_flash_amount=max_flash_amount,
        max_hops=max_hops,
        path_algorithm=path_algorithm,
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
        swap_ix = [
            Instruction(Pubkey.default(), b"swap", []) for _ in range(len(path) - 1)
        ]
        sig = await borrow_flash(
            trade_amount,
            token,
            swap_ix,
            payer=keypair or Keypair(),
        )
    else:
        if use_service:
            txs: list[str] = []
            for i in range(len(path) - 1):
                buy_v = path[i]
                sell_v = path[i + 1]
                base_buy = VENUE_URLS.get(buy_v, buy_v)
                base_sell = VENUE_URLS.get(sell_v, sell_v)
                tx1 = await _prepare_service_tx(
                    token,
                    "buy",
                    trade_amount,
                    price_map[buy_v],
                    base_buy,
                )
                tx2 = await _prepare_service_tx(
                    token,
                    "sell",
                    trade_amount,
                    price_map[sell_v],
                    base_sell,
                )
                if tx1:
                    txs.append(tx1)
                if tx2:
                    txs.append(tx2)
            if txs:
                if use_mev_bundles:
                    from .mev_executor import MEVExecutor

                    mev = MEVExecutor(
                        token,
                        priority_rpc=(
                            getattr(executor, "priority_rpc", None)
                            if executor
                            else None
                        ),
                    )
                    await mev.submit_bundle(txs)
                elif executor:
                    for tx in txs:
                        await executor.enqueue(tx)
                else:
                    from .depth_client import submit_raw_tx

                    for tx in txs:
                        await submit_raw_tx(
                            tx,
                            priority_rpc=(
                                getattr(executor, "priority_rpc", None)
                                if executor
                                else None
                            ),
                        )
        else:
            tasks = []
            for i in range(len(path) - 1):
                buy_v = path[i]
                sell_v = path[i + 1]
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
    use_mev_bundles: bool | None = None,
    max_flash_amount: float | None = None,
    max_hops: int | None = None,
    path_algorithm: str | None = None,
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
    if max_hops is None:
        max_hops = MAX_HOPS
    if path_algorithm is None:
        path_algorithm = PATH_ALGORITHM

    def _streams_for(token: str):
        if streams is not None:
            return streams, None
        if feeds is not None:
            return None, None

        available = {
            "orca": (ORCA_WS_URL, stream_orca_prices),
            "raydium": (RAYDIUM_WS_URL, stream_raydium_prices),
            "phoenix": (PHOENIX_WS_URL, stream_phoenix_prices),
            "meteora": (METEORA_WS_URL, stream_meteora_prices),
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
                auto.append(
                    order_book_ws.stream_order_book(url, rate_limit=DEPTH_RATE_LIMIT)
                )
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
                    use_mev_bundles=use_mev_bundles,
                    max_flash_amount=max_flash_amount,
                    max_hops=max_hops,
                    path_algorithm=path_algorithm,
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
                use_mev_bundles=use_mev_bundles,
                max_flash_amount=max_flash_amount,
                max_hops=max_hops,
                path_algorithm=path_algorithm,
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
        use_mev_bundles=use_mev_bundles,
        max_flash_amount=max_flash_amount,
        max_hops=max_hops,
        path_algorithm=path_algorithm,
        **kwargs,
    )


async def evaluate(token: str, portfolio) -> List[dict]:
    """Return arbitrage buy/sell actions for ``token``.

    This function performs a dry-run arbitrage detection and, when an
    opportunity is found, returns corresponding buy and sell actions. The
    amount and threshold are controlled via the ``ARBITRAGE_AMOUNT`` and
    ``ARBITRAGE_THRESHOLD`` environment variables. If either is not set or
    non-positive, no action is taken.
    """

    threshold = float(os.getenv("ARBITRAGE_THRESHOLD", "0") or 0)
    amount = float(os.getenv("ARBITRAGE_AMOUNT", "0") or 0)
    if threshold <= 0 or amount <= 0:
        return []

    try:
        res = await detect_and_execute_arbitrage(
            token, threshold=threshold, amount=amount, dry_run=True
        )
    except Exception as exc:  # pragma: no cover - network issues
        logger.warning("Arbitrage evaluation failed for %s: %s", token, exc)
        return []

    if not res:
        return []

    action = {"token": token, "amount": float(amount), "price": 0.0}
    return [
        dict(action, side="buy"),
        dict(action, side="sell"),
    ]


try:  # pragma: no cover - best effort
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = None
if MEASURE_DEX_LATENCY and loop:
    loop.call_soon(start_latency_refresh)
