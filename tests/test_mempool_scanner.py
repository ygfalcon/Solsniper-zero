import asyncio

import solhunter_zero.mempool_scanner as mp_scanner
from solhunter_zero import scanner_common


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
        {"result": {"value": {"logs": ["InitializeMint", "name: coolbonk", "mint: tok1"]}}},
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

    tokens = asyncio.run(scanner_common.offline_or_onchain_async(False, method="mempool"))
    assert tokens == ["tokmp"]


def test_stream_mempool_tokens_with_metrics(monkeypatch):
    msgs = [
        {"result": {"value": {"logs": ["InitializeMint", "name: coolbonk", "mint: tok1"]}}},
    ]

    def fake_connect(url):
        return FakeConnect(url, msgs)

    monkeypatch.setattr(mp_scanner, "connect", fake_connect)

    import solhunter_zero.onchain_metrics as om

    monkeypatch.setattr(om, "fetch_volume_onchain", lambda t, u: 1.0)
    monkeypatch.setattr(om, "fetch_liquidity_onchain", lambda t, u: 2.0)

    async def run():
        gen = mp_scanner.stream_mempool_tokens("ws://node", return_metrics=True)
        data = await asyncio.wait_for(anext(gen), timeout=0.1)
        await gen.aclose()
        return data

    data = asyncio.run(run())
    assert data == {"address": "tok1", "volume": 1.0, "liquidity": 2.0}
