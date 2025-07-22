import asyncio
import time
from solhunter_zero import arbitrage as arb
from solhunter_zero.arbitrage import detect_and_execute_arbitrage

async def dex1(token):
    return 1.0

async def dex2(token):
    return 1.2

async def dex3(token):
    return 1.4


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
