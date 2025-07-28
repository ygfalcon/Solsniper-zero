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


def test_arbitrage_path_selection(monkeypatch):
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


def test_concurrent_execution(monkeypatch):
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


def test_multi_hop(monkeypatch):
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


def test_new_venue_path(monkeypatch):
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


def test_graph_search_profit():
    prices = {"dex1": 1.0, "dex2": 1.2, "dex3": 1.3}
    old_path, old_profit = _legacy_best_route(prices, 1.0)
    new_path, new_profit = arb._best_route(
        prices,
        1.0,
        max_hops=3,
        path_algorithm="graph",
    )

    assert new_profit >= old_profit


def test_graph_vs_permutation_benchmark():
    prices = {f"dex{i}": 1.0 + 0.1 * i for i in range(6)}

    start = time.perf_counter()
    _, perm_profit = arb._best_route(
        prices,
        1.0,
        max_hops=6,
        path_algorithm="permutation",
    )
    perm_time = time.perf_counter() - start

    start = time.perf_counter()
    _, graph_profit = arb._best_route(
        prices,
        1.0,
        max_hops=6,
        path_algorithm="graph",
    )
    graph_time = time.perf_counter() - start

    assert graph_profit == pytest.approx(perm_profit)
    assert graph_time <= perm_time
