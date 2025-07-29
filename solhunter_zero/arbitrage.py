import asyncio
import logging
import os
import json
import heapq
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
from .http import get_session
from .scanner_common import JUPITER_WS_URL

from .exchange import (
    place_order_async,
    ORCA_DEX_URL,
    RAYDIUM_DEX_URL,
    DEX_BASE_URL,
    VENUE_URLS,
    SWAP_PATH,
)
from .config import load_dex_config
from . import order_book_ws
from . import depth_client
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
USE_SERVICE_ROUTE = os.getenv("USE_SERVICE_ROUTE", "0").lower() in {"1", "true", "yes"}


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


_DEX_CFG = load_dex_config()
DEX_FEES = _DEX_CFG.fees
DEX_GAS = _DEX_CFG.gas
DEX_LATENCY = _DEX_CFG.latency
EXTRA_API_URLS = {str(k): str(v) for k, v in _parse_mapping_env("DEX_API_URLS").items()}
EXTRA_WS_URLS = {str(k): str(v) for k, v in _parse_mapping_env("DEX_WS_URLS").items()}

# Flash loan configuration
USE_FLASH_LOANS = os.getenv("USE_FLASH_LOANS", "0").lower() in {"1", "true", "yes"}
MAX_FLASH_AMOUNT = float(os.getenv("MAX_FLASH_AMOUNT", "0") or 0)
USE_MEV_BUNDLES = os.getenv("USE_MEV_BUNDLES", "0").lower() in {"1", "true", "yes"}

# Path search configuration
MAX_HOPS = int(os.getenv("MAX_HOPS", "3") or 3)
PATH_ALGORITHM = os.getenv("PATH_ALGORITHM", "graph")

from .lru import LRUCache, TTLCache
from .event_bus import subscribe
from .prices import get_cached_price, update_price_cache

ROUTE_CACHE = LRUCache(maxsize=128)
_EDGE_CACHE = LRUCache(maxsize=1024)
_LAST_DEPTH: dict[str, float] = {}

# shared HTTP session and price cache
PRICE_CACHE_TTL = float(os.getenv("PRICE_CACHE_TTL", "30") or 30)
PRICE_CACHE = TTLCache(maxsize=256, ttl=PRICE_CACHE_TTL)

def _route_key(token: str, amount: float, fees: Mapping[str, float], gas: Mapping[str, float], latency: Mapping[str, float]) -> tuple:
    def _norm(m: Mapping[str, float]) -> tuple:
        return tuple(sorted((k, float(v)) for k, v in m.items()))
    return (token, float(amount), _norm(fees), _norm(gas), _norm(latency))

def invalidate_route(token: str | None = None) -> None:
    """Remove cached paths for ``token`` or clear the cache."""
    if token is None:
        ROUTE_CACHE.clear()
        return
    keys = [k for k in ROUTE_CACHE._cache if k[0] == token]
    for k in keys:
        ROUTE_CACHE._cache.pop(k, None)

def invalidate_edges(token: str | None = None) -> None:
    """Remove cached adjacency data for ``token`` or clear the cache."""
    if token is None:
        _EDGE_CACHE.clear()
        return
    _EDGE_CACHE._cache.pop(token, None)

def _on_depth_update(payload: Mapping[str, Mapping[str, float]]) -> None:
    for token, entry in payload.items():
        depth = float(entry.get("depth", 0.0))
        last = _LAST_DEPTH.get(token)
        if last is not None:
            base = max(depth, last, 1.0)
            if abs(depth - last) / base > 0.1:
                invalidate_route(token)
                invalidate_edges(token)
        _LAST_DEPTH[token] = depth

subscribe("depth_update", _on_depth_update)


def refresh_costs() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return updated fee, gas and latency mappings from the environment."""
    cfg = load_dex_config()
    return cfg.fees, cfg.gas, cfg.latency


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
    session = await get_session()
    try:
        async with session.post(
            f"{base_url}{SWAP_PATH}", json=payload, timeout=10
        ) as resp:
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
    cached = get_cached_price(token)
    if cached is not None:
        PRICE_CACHE.set(("orca", token), cached)
        return cached

    cached = PRICE_CACHE.get(("orca", token))
    if cached is not None:
        update_price_cache(token, cached)
        return cached

    url = f"{ORCA_API_URL}/price?token={token}"
    session = await get_session()
    try:
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json()
            price = data.get("price")
            value = float(price) if isinstance(price, (int, float)) else 0.0
    except aiohttp.ClientError as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch price from Orca: %s", exc)
        return 0.0

    PRICE_CACHE.set(("orca", token), value)
    update_price_cache(token, value)
    return value


async def fetch_raydium_price_async(token: str) -> float:
    """Return the current price for ``token`` from the Raydium API."""
    cached = get_cached_price(token)
    if cached is not None:
        PRICE_CACHE.set(("raydium", token), cached)
        return cached

    cached = PRICE_CACHE.get(("raydium", token))
    if cached is not None:
        update_price_cache(token, cached)
        return cached

    url = f"{RAYDIUM_API_URL}/price?token={token}"
    session = await get_session()
    try:
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json()
            price = data.get("price")
            value = float(price) if isinstance(price, (int, float)) else 0.0
    except aiohttp.ClientError as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch price from Raydium: %s", exc)
        return 0.0

    PRICE_CACHE.set(("raydium", token), value)
    update_price_cache(token, value)
    return value


async def fetch_phoenix_price_async(token: str) -> float:
    """Return the current price for ``token`` from the Phoenix API."""
    cached = get_cached_price(token)
    if cached is not None:
        PRICE_CACHE.set(("phoenix", token), cached)
        return cached

    cached = PRICE_CACHE.get(("phoenix", token))
    if cached is not None:
        update_price_cache(token, cached)
        return cached

    url = f"{PHOENIX_API_URL}/price?token={token}"
    session = await get_session()
    try:
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json()
            price = data.get("price")
            value = float(price) if isinstance(price, (int, float)) else 0.0
    except aiohttp.ClientError as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch price from Phoenix: %s", exc)
        return 0.0

    PRICE_CACHE.set(("phoenix", token), value)
    update_price_cache(token, value)
    return value


async def fetch_meteora_price_async(token: str) -> float:
    """Return the current price for ``token`` from the Meteora API."""
    cached = get_cached_price(token)
    if cached is not None:
        PRICE_CACHE.set(("meteora", token), cached)
        return cached

    cached = PRICE_CACHE.get(("meteora", token))
    if cached is not None:
        update_price_cache(token, cached)
        return cached

    url = f"{METEORA_API_URL}/price?token={token}"
    session = await get_session()
    try:
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json()
            price = data.get("price")
            value = float(price) if isinstance(price, (int, float)) else 0.0
    except aiohttp.ClientError as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch price from Meteora: %s", exc)
        return 0.0

    PRICE_CACHE.set(("meteora", token), value)
    update_price_cache(token, value)
    return value


JUPITER_API_URL = os.getenv("JUPITER_API_URL", "https://price.jup.ag/v4/price")


async def fetch_jupiter_price_async(token: str) -> float:
    """Return the current price for ``token`` from the Jupiter price API."""

    cached = get_cached_price(token)
    if cached is not None:
        PRICE_CACHE.set(("jupiter", token), cached)
        return cached

    cached = PRICE_CACHE.get(("jupiter", token))
    if cached is not None:
        update_price_cache(token, cached)
        return cached

    url = f"{JUPITER_API_URL}?ids={token}"
    session = await get_session()
    try:
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json()
            info = data.get("data", {}).get(token)
            if info and isinstance(info, dict):
                price = info.get("price")
                value = float(price) if isinstance(price, (int, float)) else 0.0
            else:
                value = 0.0
    except aiohttp.ClientError as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch price from Jupiter: %s", exc)
        return 0.0

    PRICE_CACHE.set(("jupiter", token), value)
    update_price_cache(token, value)
    return value


def make_api_price_fetch(url: str, name: str | None = None) -> PriceFeed:
    """Return a simple price fetcher for ``url``."""

    if not name:
        name = url

    async def _fetch(token: str, _name=name) -> float:
        cached = get_cached_price(token)
        if cached is not None:
            PRICE_CACHE.set((_name, token), cached)
            return cached

        cached = PRICE_CACHE.get((_name, token))
        if cached is not None:
            update_price_cache(token, cached)
            return cached

        req = f"{url.rstrip('/')}/price?token={token}"
        session = await get_session()
        try:
            async with session.get(req, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
                price = data.get("price")
                value = float(price) if isinstance(price, (int, float)) else 0.0
        except aiohttp.ClientError as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch price from %s: %s", url, exc)
            return 0.0

        PRICE_CACHE.set((_name, token), value)
        update_price_cache(token, value)
        return value

    _fetch.__name__ = str(name)
    return _fetch


async def stream_orca_prices(
    token: str, url: str = ORCA_WS_URL
) -> AsyncGenerator[float, None]:
    """Yield live prices for ``token`` from the Orca websocket feed."""

    if not url:
        return

    session = await get_session()
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


async def stream_raydium_prices(
    token: str, url: str = RAYDIUM_WS_URL
) -> AsyncGenerator[float, None]:
    """Yield live prices for ``token`` from the Raydium websocket feed."""

    if not url:
        return

    session = await get_session()
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


async def stream_phoenix_prices(
    token: str, url: str = PHOENIX_WS_URL
) -> AsyncGenerator[float, None]:
    """Yield live prices for ``token`` from the Phoenix websocket feed."""

    if not url:
        return

    session = await get_session()
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


async def stream_meteora_prices(
    token: str, url: str = METEORA_WS_URL
) -> AsyncGenerator[float, None]:
    """Yield live prices for ``token`` from the Meteora websocket feed."""

    if not url:
        return

    session = await get_session()
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


async def stream_jupiter_prices(
    token: str, url: str = JUPITER_WS_URL
) -> AsyncGenerator[float, None]:
    """Yield live prices for ``token`` from the Jupiter websocket feed."""

    if not url:
        return

    session = await get_session()
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
        session = await get_session()
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
    token: str | None = None,
    fees: Mapping[str, float] | None = None,
    gas: Mapping[str, float] | None = None,
    latency: Mapping[str, float] | None = None,
    depth: Mapping[str, Mapping[str, float]] | None = None,
    use_flash_loans: bool | None = None,
    max_flash_amount: float | None = None,
    max_hops: int | None = None,
    path_algorithm: str | None = None,
) -> tuple[list[str], float]:
    """Return path with maximum profit and the expected profit."""

    fees = fees or {}
    gas = gas or {}
    latency = latency or {}
    for v in prices.keys():
        if v not in fees:
            fees[v] = DEX_FEES.get(v, 0.0)
        if v not in gas:
            gas[v] = DEX_GAS.get(v, 0.0)
        if v not in latency:
            latency[v] = DEX_LATENCY.get(v, 0.0)
    if use_flash_loans is None:
        use_flash_loans = USE_FLASH_LOANS
    if max_flash_amount is None:
        max_flash_amount = MAX_FLASH_AMOUNT
    if max_hops is None:
        max_hops = MAX_HOPS
    if path_algorithm is None:
        path_algorithm = PATH_ALGORITHM
    if max_hops is None:
        max_hops = MAX_HOPS
    if path_algorithm is None:
        path_algorithm = PATH_ALGORITHM
    trade_amount = (
        min(max_flash_amount or amount, amount) if use_flash_loans else amount
    )
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

    def slip_cost(a: str, b: str) -> float:
        if depth is None:
            return 0.0
        a_depth = depth.get(a, {}) if isinstance(depth, Mapping) else {}
        b_depth = depth.get(b, {}) if isinstance(depth, Mapping) else {}
        ask = float(a_depth.get("asks", 0.0))
        bid = float(b_depth.get("bids", 0.0))
        slip_a = trade_amount / ask if ask > 0 else 0.0
        slip_b = trade_amount / bid if bid > 0 else 0.0
        return (
            prices[a] * trade_amount * slip_a
            + prices[b] * trade_amount * slip_b
        )

    adj_key = token if token is not None else None
    adjacency: dict[str, dict[str, float]] | None = None
    if adj_key is not None:
        adjacency = _EDGE_CACHE.get(adj_key)
    if adjacency is None:
        adjacency = {}
        for a in venues:
            neigh = {}
            for b in venues:
                if a == b:
                    continue
                profit = (
                    (prices[b] - prices[a]) * trade_amount
                    - step_cost(a, b)
                    - slip_cost(a, b)
                )
                neigh[b] = profit
            adjacency[a] = neigh
        if adj_key is not None:
            _EDGE_CACHE.set(adj_key, adjacency)

    best_path: list[str] = []
    best_profit = float("-inf")
    heap: list[tuple[float, list[str], set[str]]] = []
    for v in venues:
        heapq.heappush(heap, (0.0, [v], {v}))

    while heap:
        neg_profit, path, visited = heapq.heappop(heap)
        profit = -neg_profit
        if len(path) > 1 and profit > best_profit:
            best_profit = profit
            best_path = path
        if len(path) >= max_hops:
            continue
        last = path[-1]
        for nxt, val in adjacency.get(last, {}).items():
            if nxt in visited:
                continue
            new_profit = profit + val
            heapq.heappush(heap, (-new_profit, path + [nxt], visited | {nxt}))

    return best_path, best_profit


async def _compute_route(
    token: str,
    amount: float,
    price_map: Mapping[str, float],
    *,
    fees: Mapping[str, float],
    gas: Mapping[str, float],
    latency: Mapping[str, float],
    use_service: bool,
    use_flash_loans: bool,
    max_flash_amount: float,
    max_hops: int,
    path_algorithm: str,
) -> tuple[list[str], float]:
    key = _route_key(token, amount, fees, gas, latency)
    cached = ROUTE_CACHE.get(key)
    if cached is not None:
        return cached

    if use_service:
        res = await depth_client.best_route(
            token,
            amount,
            socket_path=DEPTH_SERVICE_SOCKET,
            max_hops=max_hops,
        )
        if res:
            path, profit, _ = res
        else:
            path, profit = [], 0.0
    else:
        depth_map, _ = depth_client.snapshot(token)
        total = sum(
            float(v.get("bids", 0.0)) + float(v.get("asks", 0.0))
            for v in depth_map.values()
        )
        _LAST_DEPTH[token] = total
        path, profit = _best_route(
            price_map,
            amount,
            token=token,
            fees=fees,
            gas=gas,
            latency=latency,
            depth=depth_map,
            use_flash_loans=use_flash_loans,
            max_flash_amount=max_flash_amount,
            max_hops=max_hops,
            path_algorithm=path_algorithm,
        )

    ROUTE_CACHE.set(key, (path, profit))
    return path, profit


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
