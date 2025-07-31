import asyncio

from solhunter_zero.agents.atlantean_wave_agent import AtlanteanWaveAgent
from solhunter_zero.portfolio import Portfolio


class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}


def test_atlantean_wave_trade(monkeypatch):
    agent = AtlanteanWaveAgent(threshold=0.3)
    monkeypatch.setattr(
        "solhunter_zero.agents.atlantean_wave_agent.pyramid_frequency", lambda s: 0.5
    )
    pf = DummyPortfolio()
    actions = asyncio.run(agent.propose_trade("TOK", pf, depth=1.0, imbalance=0.0))
    assert actions and actions[0]["side"] == "buy"
    assert actions[0]["frequency"] == 0.5


def test_atlantean_wave_no_trade(monkeypatch):
    agent = AtlanteanWaveAgent(threshold=0.6)
    monkeypatch.setattr(
        "solhunter_zero.agents.atlantean_wave_agent.pyramid_frequency", lambda s: 0.4
    )
    pf = DummyPortfolio()
    actions = asyncio.run(agent.propose_trade("TOK", pf, depth=1.0))
    assert actions == []
