import asyncio
import types
import pytest
pytest.importorskip("torch.nn.utils.rnn")

from solhunter_zero import simulation
from solhunter_zero.simulation import SimulationResult, predict_price_movement
from solhunter_zero.agents.conviction import ConvictionAgent
from solhunter_zero.agents.dqn import DQNAgent
from solhunter_zero.agents.arbitrage import ArbitrageAgent
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.memory import Memory
from solhunter_zero.portfolio import Position
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


async def _feed_low(token: str) -> float:
    return 0.6


async def _feed_high(token: str) -> float:
    return 0.8


def test_dqn_arbitrage_memory_integration(tmp_path):
    mem = Memory("sqlite:///:memory:")
    mem_agent = MemoryAgent(mem)
    dqn = DQNAgent(memory_agent=mem_agent, epsilon=0.0, model_path=tmp_path / "dqn.pt")
    pf = DummyPortfolio()

    first = asyncio.run(dqn.propose_trade("tok", pf))
    assert first and first[0]["side"] == "buy"
    asyncio.run(mem_agent.log({"token": "tok", "side": "buy", "amount": 1.0, "price": 1.0, "agent": "dqn"}))
    pf.balances["tok"] = Position("tok", 1.0, 1.0, 1.0)

    arb_agent = ArbitrageAgent(threshold=0.0, amount=1.0, feeds=[_feed_low, _feed_high])
    arb_actions = asyncio.run(arb_agent.propose_trade("tok", pf))
    assert {a["side"] for a in arb_actions} == {"buy", "sell"}
    for action in arb_actions:
        asyncio.run(mem_agent.log(action))

    second = asyncio.run(dqn.propose_trade("tok", pf))
    assert second
