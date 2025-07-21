import asyncio
import types
import pytest

from solhunter_zero import simulation
from solhunter_zero.simulation import SimulationResult, predict_price_movement
from solhunter_zero.agents.conviction import ConvictionAgent
from solhunter_zero.portfolio import Portfolio


class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}


def test_predict_price_movement_delegates(monkeypatch):
    captured = {}

    def fake_run(token, count=1, days=1, **_):
        captured['count'] = count
        captured['days'] = days
        return [SimulationResult(success_prob=1.0, expected_roi=0.12)]

    monkeypatch.setattr(simulation, 'run_simulations', fake_run)

    val = predict_price_movement('tok')
    assert val == pytest.approx(0.12)
    assert captured['count'] == 1
    assert captured['days'] == 1


def test_conviction_agent_uses_prediction(monkeypatch):
    agent = ConvictionAgent(threshold=0.05, count=1)

    monkeypatch.setattr(
        'solhunter_zero.agents.conviction.run_simulations',
        lambda t, count=1, **_: [SimulationResult(success_prob=1.0, expected_roi=0.06)],
    )
    monkeypatch.setattr('solhunter_zero.agents.conviction.predict_price_movement', lambda t: 0.04)

    actions = asyncio.run(agent.propose_trade('tok', DummyPortfolio()))
    assert actions == []
