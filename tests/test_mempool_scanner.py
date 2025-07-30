import asyncio

import solhunter_zero.mempool_scanner as mp_scanner
from solhunter_zero import scanner_common, event_bus

scanner_common.TOKEN_SUFFIX = ""


class FakeWS:
    def __init__(self, messages):
        self.messages = list(messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def logs_subscribe(self, *args, **kwargs):
        pass

    async def recv(self):
        if self.messages:
            return [self.messages.pop(0)]
        raise asyncio.CancelledError


class FakeConnect:
    def __init__(self, url, messages):
        self.url = url
        self.ws = FakeWS(messages)

    async def __aenter__(self):
        return await self.ws.__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        await self.ws.__aexit__(exc_type, exc, tb)


def test_stream_mempool_tokens(monkeypatch):
    msgs = [
        {
            "result": {
                "value": {"logs": ["InitializeMint", "name: coolbonk", "mint: tok1"]}
            }
        },
        {"result": {"value": {"logs": ["something else"]}}},
    ]

    def fake_connect(url):
        return FakeConnect(url, msgs)

    monkeypatch.setattr(mp_scanner, "connect", fake_connect)

    async def run():
        gen = mp_scanner.stream_mempool_tokens("ws://node", suffix="bonk")
        token = await asyncio.wait_for(anext(gen), timeout=0.1)
        await gen.aclose()
        return token

    token = asyncio.run(run())
    assert token == "tok1"


def test_stream_mempool_tokens_pool(monkeypatch):
    token = "A" * 36 + "BONK"
    msgs = [
        {"result": {"value": {"logs": [f"tokenA: {token}", "tokenB: x"]}}},
    ]

    def fake_connect(url):
        return FakeConnect(url, msgs)

    monkeypatch.setattr(mp_scanner, "connect", fake_connect)

    async def run():
        gen = mp_scanner.stream_mempool_tokens("ws://node", include_pools=True)
        token_out = await asyncio.wait_for(anext(gen), timeout=0.1)
        await gen.aclose()
        return token_out

    token_out = asyncio.run(run())
    assert token_out == token


def test_offline_or_onchain_async_mempool(monkeypatch):
    async def fake_stream(url, *, suffix=None, keywords=None, include_pools=True):
        yield "tokmp"

    monkeypatch.setattr(mp_scanner, "stream_mempool_tokens", fake_stream)

    scanner_common.BIRDEYE_API_KEY = None
    scanner_common.SOLANA_RPC_URL = "ws://node"

    tokens = asyncio.run(
        scanner_common.offline_or_onchain_async(False, method="mempool")
    )
    assert tokens == ["tokmp"]


def test_stream_mempool_tokens_with_metrics(monkeypatch):
    msgs = [
        {
            "result": {
                "value": {"logs": ["InitializeMint", "name: coolbonk", "mint: tok1"]}
            }
        },
    ]

    def fake_connect(url):
        return FakeConnect(url, msgs)

    monkeypatch.setattr(mp_scanner, "connect", fake_connect)

    import solhunter_zero.onchain_metrics as om

    monkeypatch.setattr(
        om, "fetch_volume_onchain", lambda t, u: 1.0
    )
    monkeypatch.setattr(
        om, "fetch_liquidity_onchain", lambda t, u: 2.0
    )

    async def run():
        gen = mp_scanner.stream_mempool_tokens("ws://node", return_metrics=True)
        data = await asyncio.wait_for(anext(gen), timeout=0.1)
        await gen.aclose()
        return data

    data = asyncio.run(run())
    assert data == {"address": "tok1", "volume": 1.0, "liquidity": 2.0}


def test_stream_ranked_mempool_tokens(monkeypatch):
    msgs = [
        {"result": {"value": {"logs": ["InitializeMint", "name: tok1", "mint: tok1"]}}},
        {"result": {"value": {"logs": ["InitializeMint", "name: tok2", "mint: tok2"]}}},
    ]

    def fake_connect(url):
        return FakeConnect(url, msgs)

    monkeypatch.setattr(mp_scanner, "connect", fake_connect)

    import solhunter_zero.onchain_metrics as om

    monkeypatch.setattr(
        om,
        "fetch_volume_onchain_async",
        lambda t, u: asyncio.sleep(0, 10.0 if t == "tok1" else 1.0),
    )
    monkeypatch.setattr(
        om,
        "fetch_liquidity_onchain_async",
        lambda t, u: asyncio.sleep(0, 5.0 if t == "tok1" else 0.5),
    )
    monkeypatch.setattr(
        om,
        "collect_onchain_insights_async",
        lambda t, u: asyncio.sleep(
            0,
            {
                "tx_rate": 2.0 if t == "tok1" else 0.1,
                "whale_activity": 0.0,
                "avg_swap_size": 1.0,
            },
        ),
    )

    async def run():
        gen = mp_scanner.stream_ranked_mempool_tokens(
            "ws://node", suffix="", threshold=10.0
        )
        data = await asyncio.wait_for(anext(gen), timeout=0.1)
        await gen.aclose()
        return data

    data = asyncio.run(run())
    assert data["address"] == "tok1"
    assert data["score"] >= 10.0
    expected = data["momentum"] * (1.0 - data["whale_activity"])
    assert data["combined_score"] == expected


def test_rank_token_momentum(monkeypatch):
    mp_scanner._ROLLING_STATS.clear()
    import solhunter_zero.onchain_metrics as om

    monkeypatch.setattr(
        om, "fetch_volume_onchain_async", lambda t, u: asyncio.sleep(0, 1.0)
    )
    monkeypatch.setattr(
        om, "fetch_liquidity_onchain_async", lambda t, u: asyncio.sleep(0, 1.0)
    )
    rates = [1.0, 3.0]

    def fake_insights(t, u):
        return {"tx_rate": rates.pop(0), "whale_activity": 0.0, "avg_swap_size": 1.0}

    async def fake_insights_async(t, u):
        return fake_insights(t, u)

    monkeypatch.setattr(om, "collect_onchain_insights_async", fake_insights_async)

    async def run():
        first = await mp_scanner.rank_token("tok", "rpc")
        second = await mp_scanner.rank_token("tok", "rpc")
        return first, second

    first, second = asyncio.run(run())
    assert second[1]["momentum"] != 0.0


def test_stream_ranked_with_depth(monkeypatch):
    async def fake_gen(url, **_):
        yield {
            "address": "tok1",
            "combined_score": 1.0,
            "momentum": 1.0,
            "whale_activity": 0.0,
        }
        yield {
            "address": "tok2",
            "combined_score": 1.0,
            "momentum": 1.0,
            "whale_activity": 0.0,
        }

    monkeypatch.setattr(mp_scanner, "stream_ranked_mempool_tokens", fake_gen)

    import solhunter_zero.order_book_ws as obws

    monkeypatch.setattr(
        obws, "snapshot", lambda t: (5.0 if t == "tok1" else 10.0, 0.0, 0.0)
    )

    async def run():
        gen = mp_scanner.stream_ranked_mempool_tokens_with_depth("rpc")
        r1 = await asyncio.wait_for(anext(gen), timeout=0.1)
        r2 = await asyncio.wait_for(anext(gen), timeout=0.1)
        await gen.aclose()
        return r1, r2

    first, second = asyncio.run(run())
    assert second["combined_score"] > first["combined_score"]


def test_default_concurrency(monkeypatch):
    monkeypatch.setattr(mp_scanner.os, "cpu_count", lambda: 4)
    mp_scanner._CPU_PERCENT = 0.0

    async def fake_stream(_url, **__):
        for t in ("a", "b", "c", "d"):
            yield t

    monkeypatch.setattr(mp_scanner, "stream_mempool_tokens", fake_stream)

    running = 0
    max_running = 0

    async def fake_rank(_t, _u):
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        await asyncio.sleep(0)
        running -= 1
        return 0.0, {"whale_activity": 0.0, "momentum": 0.0}

    monkeypatch.setattr(mp_scanner, "rank_token", fake_rank)

    async def run():
        gen = mp_scanner.stream_ranked_mempool_tokens("rpc")
        async for _ in gen:
            pass

    asyncio.run(run())
    assert max_running <= 4


def test_cpu_threshold_reduces_concurrency(monkeypatch):
    monkeypatch.setattr(mp_scanner.os, "cpu_count", lambda: 4)
    mp_scanner._CPU_PERCENT = 90.0

    orig_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 0.05:
            event_bus.publish("resource_update", {"cpu": 10.0})
        await orig_sleep(0)

    monkeypatch.setattr(mp_scanner.asyncio, "sleep", fake_sleep)

    async def fake_stream(_url, **__):
        for t in ("a", "b"):
            yield t

    monkeypatch.setattr(mp_scanner, "stream_mempool_tokens", fake_stream)

    running = 0
    max_running = 0

    async def fake_rank(_t, _u):
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        await asyncio.sleep(0)
        running -= 1
        return 0.0, {"whale_activity": 0.0, "momentum": 0.0}

    monkeypatch.setattr(mp_scanner, "rank_token", fake_rank)

    async def run():
        gen = mp_scanner.stream_ranked_mempool_tokens("rpc", cpu_usage_threshold=80.0)
        async for _ in gen:
            pass

    asyncio.run(run())
    assert max_running <= 2
