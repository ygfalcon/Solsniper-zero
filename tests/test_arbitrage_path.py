import asyncio
import time
import pytest
from solhunter_zero import arbitrage as arb
from solhunter_zero.arbitrage import detect_and_execute_arbitrage
from itertools import permutations

async def dex1(token):
    return 1.0

async def dex2(token):
    return 1.2

async def dex3(token):
    return 1.4

async def phoenix(token):
    return 0.9

async def meteora(token):
    return 1.6


@pytest.fixture(autouse=True)
def _disable_jup(monkeypatch):
    monkeypatch.setattr(arb, "JUPITER_WS_URL", "")
    monkeypatch.setattr(arb, "USE_DEPTH_STREAM", False)
    monkeypatch.setattr(arb, "USE_SERVICE_EXEC", False)
    arb.invalidate_route()
    arb.invalidate_edges()


@pytest.fixture(params=[False, True])
def numba_enabled(monkeypatch, request):
    monkeypatch.setattr(arb, "USE_NUMBA_ROUTE", request.param)
    return request.param


@pytest.fixture(params=[False, True])
def ffi_enabled(monkeypatch, request):
    monkeypatch.setenv("USE_FFI_ROUTE", "1" if request.param else "0")
    if request.param:
        from pathlib import Path
        import subprocess
        lib_path = Path(__file__).resolve().parents[1] / "route_ffi/target/release/libroute_ffi.so"
        if not lib_path.exists():
            subprocess.run([
                "cargo",
                "build",
                "--manifest-path",
                str(Path(__file__).resolve().parents[1] / "route_ffi/Cargo.toml"),
                "--release",
            ], check=True)
        monkeypatch.setenv("ROUTE_FFI_LIB", str(lib_path))
    import importlib
    importlib.reload(arb._routeffi)
    importlib.reload(arb)
    return request.param


def test_arbitrage_path_selection(monkeypatch, numba_enabled, ffi_enabled):
    calls = []

    async def fake_place(token, side, amount, price, **_):
        calls.append((side, price))
        return {"ok": True}

    monkeypatch.setattr(arb, "place_order_async", fake_place)

    result = asyncio.run(
        detect_and_execute_arbitrage(
            "tok",
            [dex1, dex2, dex3],
            threshold=0.0,
            amount=1.0,
            fees={"dex3": 0.5},
            gas={"dex1": 0.0, "dex2": 0.0, "dex3": 0.0},
            latencies={"dex1": 0.0, "dex2": 0.0, "dex3": 0.0},
        )
    )

    assert result == (0, 1)
    assert ("sell", 1.2) in calls


def test_concurrent_execution(monkeypatch, numba_enabled, ffi_enabled):
    async def slow_place(*a, **k):
        await asyncio.sleep(0.05)
        return {"ok": True}

    monkeypatch.setattr(arb, "place_order_async", slow_place)

    start = time.perf_counter()
    asyncio.run(
        detect_and_execute_arbitrage("tok", [dex1, dex2], threshold=0.0, amount=1.0)
    )
    duration = time.perf_counter() - start
    assert duration < 0.1


def test_multi_hop(monkeypatch, numba_enabled, ffi_enabled):
    calls = []

    async def fake_place(token, side, amount, price, **_):
        calls.append((side, price))
        return {"ok": True}

    monkeypatch.setattr(arb, "place_order_async", fake_place)

    result = asyncio.run(
        detect_and_execute_arbitrage(
            "tok",
            [dex1, dex2, dex3],
            threshold=0.0,
            amount=1.0,
            fees={"dex1": 0.0, "dex2": 0.0, "dex3": 0.0},
            gas={"dex1": 0.0, "dex2": 0.0, "dex3": 0.0},
            latencies={"dex1": 0.0, "dex2": -0.2, "dex3": 0.0},
        )
    )

    assert result == (0, 2)
    assert len(calls) == 4


def test_new_venue_path(monkeypatch, numba_enabled, ffi_enabled):
    calls = []

    async def fake_place(token, side, amount, price, **_):
        calls.append((side, price))
        return {"ok": True}

    monkeypatch.setattr(arb, "place_order_async", fake_place)

    feeds = [phoenix, dex1, dex2, meteora]
    costs = {"phoenix": 0.0, "dex1": 0.0, "dex2": 0.0, "meteora": 0.0}

    result = asyncio.run(
        detect_and_execute_arbitrage(
            "tok",
            feeds,
            threshold=0.0,
            amount=1.0,
            fees=costs,
            gas=costs,
            latencies=costs,
        )
    )

    assert result == (0, 3)
    assert ("buy", 0.9) in calls
    assert ("sell", 1.6) in calls


def _legacy_best_route(prices, amount):
    fees = {k: 0.0 for k in prices}
    gas = {k: 0.0 for k in prices}
    latency = {k: 0.0 for k in prices}
    trade_amount = amount
    venues = list(prices.keys())
    best, best_profit = [], float("-inf")

    def step_cost(a, b):
        return (
            prices[a] * trade_amount * fees[a]
            + prices[b] * trade_amount * fees[b]
            + gas[a]
            + gas[b]
            + latency[a]
            + latency[b]
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
    return best, best_profit


def test_graph_search_profit(numba_enabled, ffi_enabled):
    prices = {"dex1": 1.0, "dex2": 1.2, "dex3": 1.3}
    old_path, old_profit = _legacy_best_route(prices, 1.0)
    new_path, new_profit = arb._best_route(
        prices,
        1.0,
        token="tok",
        max_hops=3,
    )

    assert new_profit >= old_profit


def test_weighted_algorithm_profit(numba_enabled, ffi_enabled):
    prices = {f"dex{i}": 1.0 + 0.1 * i for i in range(6)}

    start = time.perf_counter()
    _, p1 = arb._best_route(
        prices,
        1.0,
        token="tok",
        max_hops=6,
    )
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    _, p2 = arb._best_route(
        prices,
        1.0,
        token="tok",
        max_hops=6,
        path_algorithm="dijkstra",
    )
    t2 = time.perf_counter() - start

    assert p1 == pytest.approx(p2)
    assert t2 <= t1 * 1.5


class _CountingCache(arb.TTLCache):
    def __init__(self, ttl=60.0):
        super().__init__(maxsize=128, ttl=ttl)
        self.set_count = 0

    def set(self, key, value):
        self.set_count += 1
        super().set(key, value)


def test_edge_cache_reuse(monkeypatch, numba_enabled, ffi_enabled):
    cache = _CountingCache()
    monkeypatch.setattr(arb, "_EDGE_CACHE", cache)

    prices = {"dex1": 1.0, "dex2": 1.2, "dex3": 1.3}
    depth = {v: {"bids": 1.0, "asks": 1.0} for v in prices}

    arb._best_route(prices, 1.0, token="tok", depth=depth)
    first = cache.set_count

    arb._best_route(prices, 1.0, token="tok", depth=depth)
    assert cache.set_count == first


def test_edge_cache_invalidation(monkeypatch, numba_enabled, ffi_enabled):
    cache = _CountingCache(ttl=0.05)
    monkeypatch.setattr(arb, "_EDGE_CACHE", cache)

    prices = {"dex1": 1.0, "dex2": 1.2}
    depth = {v: {"bids": 1.0, "asks": 1.0} for v in prices}

    arb._best_route(prices, 1.0, token="tok", depth=depth)
    first = cache.set_count
    time.sleep(0.06)
    arb._best_route(prices, 1.0, token="tok", depth=depth)
    assert cache.set_count > first
