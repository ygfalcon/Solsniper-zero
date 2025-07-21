import asyncio
import json
import aiohttp
import pytest

from solhunter_zero import order_book_ws
from solhunter_zero.agents.conviction import ConvictionAgent
from solhunter_zero.agents.arbitrage import ArbitrageAgent
from solhunter_zero.simulation import SimulationResult
from solhunter_zero.portfolio import Portfolio


class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}


class FakeMsg:
    def __init__(self, data):
        self.type = aiohttp.WSMsgType.TEXT
        self.data = json.dumps(data)


class FakeWS:
    def __init__(self, messages):
        self.messages = list(messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.messages:
            return FakeMsg(self.messages.pop(0))
        raise StopAsyncIteration


class FakeSession:
    def __init__(self, messages):
        self.messages = messages

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def ws_connect(self, url):
        self.url = url
        return FakeWS(self.messages)


def test_stream_order_book(monkeypatch):
    msgs = [{"token": "tok", "bids": 60, "asks": 40}]
    monkeypatch.setattr("aiohttp.ClientSession", lambda: FakeSession(msgs))

    async def run():
        gen = order_book_ws.stream_order_book("ws://dex", rate_limit=0, max_updates=1)
        data = await anext(gen)
        await gen.aclose()
        return data

    res = asyncio.run(run())
    assert res["depth"] == 100
    assert res["imbalance"] == pytest.approx(0.2)


def test_agents_use_depth(monkeypatch):
    agent = ConvictionAgent(threshold=0.05, count=1)

    captured = {}

    def fake_run(token, count=1, order_book_strength=None, **_):
        captured["obs"] = order_book_strength
        return [SimulationResult(1.0, 0.06)]

    monkeypatch.setattr(
        "solhunter_zero.agents.conviction.run_simulations", fake_run
    )
    monkeypatch.setattr(
        "solhunter_zero.agents.conviction.predict_price_movement", lambda t: 0.0
    )

    pf = DummyPortfolio()
    actions = asyncio.run(
        agent.propose_trade("tok", pf, depth=100.0, imbalance=0.5)
    )
    assert actions and actions[0]["side"] == "buy"
    assert captured["obs"] == 100.0

    async def feed_low(token):
        return 1.0

    async def feed_high(token):
        return 1.2

    arb = ArbitrageAgent(threshold=0.1, amount=5, feeds=[feed_low, feed_high])
    no_trade = asyncio.run(
        arb.propose_trade("tok", pf, depth=-1.0, imbalance=0.0)
    )
    assert no_trade == []

