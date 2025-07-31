import asyncio
from solhunter_zero.agents.crossdex_rebalancer import CrossDEXRebalancer
from solhunter_zero.agents.portfolio_optimizer import PortfolioOptimizer
from solhunter_zero.agents.execution import ExecutionAgent
from solhunter_zero.portfolio import Portfolio


class DummyExec(ExecutionAgent):
    def __init__(self):
        super().__init__(rate_limit=0)
        self.actions = []

    async def execute(self, action):
        self.actions.append(action)
        return action


def test_crossdex_split(monkeypatch):
    pf = Portfolio(path=None)

    async def fake_opt(self, token, pf, *, depth=None, imbalance=None):
        return [{"token": token, "side": "buy", "amount": 10.0, "price": 1.0}]

    monkeypatch.setattr(
        PortfolioOptimizer,
        "propose_trade",
        fake_opt,
    )

    monkeypatch.setattr(
        "solhunter_zero.agents.crossdex_rebalancer.snapshot",
        lambda t: ({"dexA": {"bids": 50, "asks": 100}, "dexB": {"bids": 50, "asks": 20}}, 0.0),
    )

    async def fake_latency(urls):
        return {"dexA": 0.5, "dexB": 0.01}

    monkeypatch.setattr(
        "solhunter_zero.agents.crossdex_rebalancer.measure_dex_latency_async",
        fake_latency,
    )
    monkeypatch.setattr(
        "solhunter_zero.agents.crossdex_rebalancer.DEX_FEES", {"dexA": 0.0, "dexB": 0.0}
    )

    exec_agent = DummyExec()
    agent = CrossDEXRebalancer(
        executor=exec_agent,
        rebalance_interval=0,
        slippage_threshold=1.0,
    )
    asyncio.run(agent.propose_trade("tok", pf))

    assert exec_agent.actions
    assert exec_agent.actions[0]["venue"] == "dexB"


def test_crossdex_feed(monkeypatch):
    pf = Portfolio(path=None)

    async def fake_opt(self, token, pf, *, depth=None, imbalance=None):
        return [{"token": token, "side": "buy", "amount": 10.0, "price": 1.0}]

    monkeypatch.setattr(
        PortfolioOptimizer,
        "propose_trade",
        fake_opt,
    )

    def fail_snapshot(t):
        raise AssertionError("snapshot called")

    monkeypatch.setattr(
        "solhunter_zero.agents.crossdex_rebalancer.snapshot",
        fail_snapshot,
    )

    async def fake_latency(urls):
        return {"dexA": 0.1, "dexB": 0.1}

    monkeypatch.setattr(
        "solhunter_zero.agents.crossdex_rebalancer.measure_dex_latency_async",
        fake_latency,
    )
    monkeypatch.setattr(
        "solhunter_zero.agents.crossdex_rebalancer.DEX_FEES", {"dexA": 0.0, "dexB": 0.0}
    )

    monkeypatch.setenv("USE_DEPTH_FEED", "True")
    exec_agent = DummyExec()
    agent = CrossDEXRebalancer(executor=exec_agent, rebalance_interval=0)

    from solhunter_zero.event_bus import publish

    publish(
        "depth_update",
        {"tok": {"dexA": {"bids": 50, "asks": 100}, "dexB": {"bids": 50, "asks": 20}}},
    )

    asyncio.run(agent.propose_trade("tok", pf))
    agent.close()

    assert exec_agent.actions
    assert exec_agent.actions[0]["venue"] == "dexA"
