"""Routing utilities and FFI integrations for arbitrage."""
from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
import os
from typing import Mapping, Sequence, Tuple, Optional, List

import numpy as np

from solhunter_zero.lru import LRUCache, TTLCache

from ..config import load_dex_config
from ..event_bus import subscribe
from .. import depth_client
from ..exchange import VENUE_URLS

try:  # optional rust ffi
    from .. import routeffi as _routeffi

    _HAS_ROUTEFFI = _routeffi.available()
    _HAS_PARALLEL = _routeffi.parallel_enabled()
except Exception:  # pragma: no cover - ffi unavailable
    _HAS_ROUTEFFI = False
    _HAS_PARALLEL = False
    _routeffi = None  # type: ignore

_ffi_env = os.getenv("USE_FFI_ROUTE")
if _ffi_env is not None:
    USE_FFI_ROUTE = _ffi_env.strip().lower() not in {"0", "false", "no"}
else:
    try:
        USE_FFI_ROUTE = _routeffi.available()
    except Exception:
        USE_FFI_ROUTE = False

try:  # pragma: no cover - optional dependency
    from numba import njit as _numba_njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba not available
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        if args and callable(args[0]):
            return args[0]

        def wrapper(func):
            return func

        return wrapper
else:  # pragma: no cover - numba available

    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            try:
                return _numba_njit(*args, **kwargs)(func)
            except Exception:
                return func

        if args and callable(args[0]):
            return decorator(args[0])

        return decorator

logger = logging.getLogger(__name__)


def _parse_mapping_env(env: str) -> dict:
    """Return dictionary from ``env`` or an empty mapping."""
    val = os.getenv(env)
    if not val:
        return {}
    try:
        import orjson as _orjson

        data = _orjson.loads(val)
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
FLASH_LOAN_RATIO = float(os.getenv("FLASH_LOAN_RATIO", "0") or 0)
USE_MEV_BUNDLES = os.getenv("USE_MEV_BUNDLES", "0").lower() in {"1", "true", "yes"}

# Path search configuration
MAX_HOPS = int(os.getenv("MAX_HOPS", "3") or 3)
PATH_ALGORITHM = os.getenv("PATH_ALGORITHM", "graph")
USE_NUMBA_ROUTE = os.getenv("USE_NUMBA_ROUTE", "0").lower() in {"1", "true", "yes"}
USE_GNN_ROUTING = os.getenv("USE_GNN_ROUTING", "0").lower() in {"1", "true", "yes"}
GNN_MODEL_PATH = os.getenv("GNN_MODEL_PATH", "route_gnn.pt")
ROUTE_GENERATOR_PATH = os.getenv("ROUTE_GENERATOR_PATH", "route_generator.pt")

ROUTE_CACHE = LRUCache(maxsize=128)
EDGE_CACHE_TTL = float(os.getenv("EDGE_CACHE_TTL", "60") or 60)
_EDGE_CACHE = TTLCache(maxsize=1024, ttl=EDGE_CACHE_TTL)
_LAST_DEPTH: dict[str, float] = {}
MEMPOOL_WEIGHT = float(os.getenv("MEMPOOL_WEIGHT", "0.0001") or 0.0001)


def _route_key(
    token: str,
    amount: float,
    fees: Mapping[str, float],
    gas: Mapping[str, float],
    latency: Mapping[str, float],
) -> tuple:
    def _norm(m: Mapping[str, float]) -> tuple:
        return tuple(sorted((k, float(v)) for k, v in m.items()))

    return (token, float(amount), _norm(fees), _norm(gas), _norm(latency))


def invalidate_route(token: str | None = None) -> None:
    """Remove cached paths for ``token`` or clear the cache."""
    if token is None:
        ROUTE_CACHE.clear()
        return
    keys = [k for k in ROUTE_CACHE.keys() if k[0] == token]
    for k in keys:
        ROUTE_CACHE.pop(k, None)


def invalidate_edges(token: str | None = None) -> None:
    """Remove cached adjacency data for ``token`` or clear the cache."""
    if token is None:
        _EDGE_CACHE.clear()
        return
    _EDGE_CACHE.pop(token, None)


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


def refresh_costs() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return updated fee, gas and latency mappings from the environment."""
    cfg = load_dex_config()
    latency = dict(cfg.latency)
    if os.getenv("MEASURE_DEX_LATENCY", "1").lower() not in {"0", "false", "no"}:
        try:
            from .latency import measure_dex_latency

            latency.update(measure_dex_latency())
        except Exception as exc:  # pragma: no cover - measurement failures
            logger.debug("DEX latency measurement failed: %s", exc)
    global DEX_FEES, DEX_GAS, DEX_LATENCY
    DEX_FEES = cfg.fees
    DEX_GAS = cfg.gas
    DEX_LATENCY = latency
    return DEX_FEES, DEX_GAS, DEX_LATENCY


def _build_adjacency(
    prices: Mapping[str, float],
    trade_amount: float,
    fees: Mapping[str, float],
    gas: Mapping[str, float],
    latency: Mapping[str, float],
    depth: Mapping[str, Mapping[str, float]] | None,
    token: str | None,
    mempool_rate: float = 0.0,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Return adjacency map for the search graph with caching."""

    venues = list(prices.keys())
    adj_key = token if token is not None else None
    adjacency: dict[str, dict[str, float]] | None = None
    if adj_key is not None:
        adjacency = _EDGE_CACHE.get(adj_key)
    if adjacency is None:
        service = depth_client.get_adjacency_matrix(token) if token else None
        if service and set(service[0]) == set(venues):
            venues = service[0]
            coeff = np.array(service[1], dtype=float)
            price_arr = np.array([prices[v] for v in venues], dtype=float)
            fee_arr = np.array([fees.get(v, 0.0) for v in venues], dtype=float)
            gas_arr = np.array([gas.get(v, 0.0) for v in venues], dtype=float)
            lat_arr = np.array([latency.get(v, 0.0) for v in venues], dtype=float)
            base_cost = (
                price_arr * trade_amount * fee_arr
                + gas_arr
                + lat_arr
                + mempool_rate * MEMPOOL_WEIGHT
            )
            step_matrix = base_cost[:, None] + base_cost[None, :]
            profit_matrix = coeff * trade_amount - step_matrix
            np.fill_diagonal(profit_matrix, float("-inf"))
            adjacency = {
                a: {
                    b: float(profit_matrix[i, j])
                    for j, b in enumerate(venues)
                    if i != j
                }
                for i, a in enumerate(venues)
            }
        else:
            adjacency = {v: {} for v in venues}
        if adj_key is not None:
            _EDGE_CACHE.set(adj_key, adjacency)
    return venues, adjacency


def _list_paths(
    prices: Mapping[str, float],
    amount: float,
    *,
    token: str | None = None,
    fees: Mapping[str, float] | None = None,
    gas: Mapping[str, float] | None = None,
    latency: Mapping[str, float] | None = None,
    depth: Mapping[str, Mapping[str, float]] | None = None,
    mempool_rate: float = 0.0,
    use_flash_loans: bool | None = None,
    max_flash_amount: float | None = None,
    max_hops: int | None = None,
    top_k: int = 5,
) -> list[tuple[list[str], float]]:
    """Return top ``top_k`` paths ranked by computed profit."""

    fees = fees or {}
    gas = gas or {}
    latency = latency or {}
    for v in prices.keys():
        fees.setdefault(v, DEX_FEES.get(v, 0.0))
        gas.setdefault(v, DEX_GAS.get(v, 0.0))
        latency.setdefault(v, DEX_LATENCY.get(v, 0.0))
    if use_flash_loans is None:
        use_flash_loans = USE_FLASH_LOANS
    if max_flash_amount is None:
        max_flash_amount = MAX_FLASH_AMOUNT
    if max_hops is None:
        max_hops = MAX_HOPS

    trade_amount = (
        min(max_flash_amount or amount, amount) if use_flash_loans else amount
    )

    venues, adjacency = _build_adjacency(
        prices,
        trade_amount,
        fees,
        gas,
        latency,
        depth,
        token,
        mempool_rate,
    )

    heap: list[tuple[float, list[str], set[str]]] = []
    for v in venues:
        heapq.heappush(heap, (0.0, [v], {v}))

    paths: list[tuple[list[str], float]] = []
    while heap:
        neg_profit, path, visited = heapq.heappop(heap)
        profit = -neg_profit
        if len(path) > 1:
            paths.append((path, profit))
        if len(path) >= max_hops:
            continue
        last = path[-1]
        for nxt, val in adjacency.get(last, {}).items():
            if nxt in visited:
                continue
            new_profit = profit + val
            heapq.heappush(heap, (-new_profit, path + [nxt], visited | {nxt}))

    paths.sort(key=lambda p: p[1], reverse=True)
    return paths[:top_k]


def _best_route_py(
    prices: Mapping[str, float],
    amount: float,
    *,
    token: str | None = None,
    fees: Mapping[str, float] | None = None,
    gas: Mapping[str, float] | None = None,
    latency: Mapping[str, float] | None = None,
    depth: Mapping[str, Mapping[str, float]] | None = None,
    mempool_rate: float = 0.0,
    use_flash_loans: bool | None = None,
    max_flash_amount: float | None = None,
    max_hops: int | None = None,
    path_algorithm: str | None = None,
) -> tuple[list[str], float]:
    """Return path with maximum profit and the expected profit."""

    paths = _list_paths(
        prices,
        amount,
        token=token,
        fees=fees,
        gas=gas,
        latency=latency,
        depth=depth,
        mempool_rate=mempool_rate,
        use_flash_loans=use_flash_loans,
        max_flash_amount=max_flash_amount,
        max_hops=max_hops,
    )
    return paths[0] if paths else ([], float("-inf"))


def _best_route_numba(
    prices: Mapping[str, float],
    amount: float,
    *,
    token: str | None = None,
    fees: Mapping[str, float] | None = None,
    gas: Mapping[str, float] | None = None,
    latency: Mapping[str, float] | None = None,
    depth: Mapping[str, Mapping[str, float]] | None = None,
    mempool_rate: float = 0.0,
    use_flash_loans: bool | None = None,
    max_flash_amount: float | None = None,
    max_hops: int | None = None,
    path_algorithm: str | None = None,
) -> tuple[list[str], float]:
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
    trade_amount = (
        min(max_flash_amount or amount, amount) if use_flash_loans else amount
    )

    venues, adjacency = _build_adjacency(
        prices,
        trade_amount,
        fees,
        gas,
        latency,
        depth,
        token,
        mempool_rate,
    )
    index = {v: i for i, v in enumerate(venues)}
    import numpy as np

    n = len(venues)
    matrix = np.full((n, n), float("-inf"), dtype=float)
    for a, neigh in adjacency.items():
        i = index[a]
        for b, val in neigh.items():
            j = index[b]
            matrix[i, j] = float(val)

    idx_path, profit = _search_numba(matrix, int(max_hops))
    path = [venues[i] for i in idx_path]
    return path, float(profit)


def _best_route(
    prices: Mapping[str, float],
    amount: float,
    *,
    mempool_rate: float = 0.0,
    use_gnn_routing: bool | None = None,
    gnn_model_path: str | None = None,
    **kwargs,
) -> tuple[list[str], float]:
    """Return the best route using the Rust FFI when available."""

    if use_gnn_routing is None:
        use_gnn_routing = USE_GNN_ROUTING
    if gnn_model_path is None:
        gnn_model_path = GNN_MODEL_PATH

    if use_gnn_routing:
        try:
            from .models.gnn import load_route_gnn, rank_routes
            from .models.route_generator import load_route_generator

            model = load_route_gnn(gnn_model_path)
            generator = load_route_generator(ROUTE_GENERATOR_PATH)
        except Exception:
            model = None
            generator = None
        if model is not None:
            routes: list[list[str]] | None = None
            if generator is not None:
                try:
                    routes = generator.generate(
                        max_hops=kwargs.get("max_hops", MAX_HOPS)
                    )
                except Exception:
                    routes = None
            cand: list[tuple[list[str], float]]
            if routes:
                fees = kwargs.get("fees")
                gas = kwargs.get("gas")
                latency = kwargs.get("latency")
                depth = kwargs.get("depth")
                token = kwargs.get("token")
                use_flash_loans = kwargs.get("use_flash_loans")
                max_flash_amount = kwargs.get("max_flash_amount")
                trade_amount = (
                    min(max_flash_amount or amount, amount)
                    if (use_flash_loans or USE_FLASH_LOANS)
                    else amount
                )
                venues, adjacency = _build_adjacency(
                    prices,
                    trade_amount,
                    fees or {},
                    gas or {},
                    latency or {},
                    depth,
                    token,
                    mempool_rate,
                )
                cand = []
                for r in routes:
                    profit = 0.0
                    valid = True
                    for a, b in zip(r[:-1], r[1:]):
                        val = adjacency.get(a, {}).get(b)
                        if val is None:
                            valid = False
                            break
                        profit += val
                    if valid:
                        cand.append((r, profit))
            else:
                cand = _list_paths(
                    prices,
                    amount,
                    mempool_rate=mempool_rate,
                    **kwargs,
                )
            if cand:
                idx = rank_routes(model, [p for p, _ in cand])
                return cand[idx]
    path_algo = kwargs.get("path_algorithm")
    if path_algo == "dijkstra":
        return _best_route_numba(
            prices,
            amount,
            mempool_rate=mempool_rate,
            **kwargs,
        )
    if USE_FFI_ROUTE and _routeffi.available():
        fees = dict(kwargs.get("fees") or {})
        gas = dict(kwargs.get("gas") or {})
        latency = dict(kwargs.get("latency") or {})
        for v in prices.keys():
            fees.setdefault(v, DEX_FEES.get(v, 0.0))
            gas.setdefault(v, DEX_GAS.get(v, 0.0))
            latency.setdefault(v, DEX_LATENCY.get(v, 0.0))
        ffi_kwargs = {
            "fees": fees,
            "gas": gas,
            "latency": latency,
            "max_hops": kwargs.get("max_hops", MAX_HOPS),
        }
        try:
            if _routeffi.parallel_enabled():
                func = _routeffi.best_route_parallel
            else:
                func = _routeffi.best_route
            res = func(dict(prices), amount, **ffi_kwargs)
            if res:
                return res
        except Exception as exc:  # pragma: no cover - optional ffi failures
            logger.warning("ffi.best_route failed: %s", exc)
    return _best_route_py(prices, amount, mempool_rate=mempool_rate, **kwargs)


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

    res = None
    if use_service:
        try:
            res = await depth_client.cached_route(
                token,
                amount,
                socket_path=os.getenv("DEPTH_SERVICE_SOCKET", "/tmp/depth_service.sock"),
                max_hops=max_hops,
            )
        except Exception as exc:  # pragma: no cover - service optional
            logger.debug("depth_service.cached_route failed: %s", exc)
            res = None
        if not res:
            try:
                res = await depth_client.best_route(
                    token,
                    amount,
                    socket_path=os.getenv("DEPTH_SERVICE_SOCKET", "/tmp/depth_service.sock"),
                    max_hops=max_hops,
                )
            except Exception as exc:  # pragma: no cover - service optional
                logger.warning("depth_service.best_route failed: %s", exc)
                res = None
        if res:
            path, profit, _ = res
        else:
            use_service = False
    if not use_service or not res:
        depth_map, mempool_rate = depth_client.snapshot(token)
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
            mempool_rate=mempool_rate,
            use_flash_loans=use_flash_loans,
            max_flash_amount=max_flash_amount,
            max_hops=max_hops,
            path_algorithm=path_algorithm,
        )

    ROUTE_CACHE[key] = (path, profit)
    return path, profit


subscribe("depth_update", _on_depth_update)
