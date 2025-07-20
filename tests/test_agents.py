import asyncio
import types

import pytest

from solhunter_zero.agents.simulation import SimulationAgent
from solhunter_zero.agents.conviction import ConvictionAgent
from solhunter_zero.agents.arbitrage import ArbitrageAgent
from solhunter_zero.agents.exit import ExitAgent
from solhunter_zero.agents.execution import ExecutionAgent
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.agent_manager import AgentManager
from solhunter_zero.portfolio import Portfolio, Position


class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}


def test_simulation_agent_buy(monkeypatch):
    agent = SimulationAgent(count=1)

    monkeypatch.setattr(
        'solhunter_zero.agents.simulation.run_simulations',
        lambda t, count=1: [types.SimpleNamespace(expected_roi=0.5)],
    )
    monkeypatch.setattr(
        'solhunter_zero.agents.simulation.should_buy', lambda sims: True
    )
    monkeypatch.setattr(
        'solhunter_zero.agents.simulation.should_sell', lambda sims: False
    )
    async def fake_prices(tokens):
        return {next(iter(tokens)): 1.0}

    monkeypatch.setattr(
        'solhunter_zero.agents.simulation.fetch_token_prices_async',
        fake_prices,
    )

    actions = asyncio.run(agent.propose_trade('tok', DummyPortfolio()))
    assert actions and actions[0]['side'] == 'buy'


def test_conviction_agent_threshold(monkeypatch):
    agent = ConvictionAgent(threshold=0.1, count=1)
    monkeypatch.setattr(
        'solhunter_zero.agents.conviction.run_simulations',
        lambda t, count=1: [types.SimpleNamespace(expected_roi=0.2)],
    )

    actions = asyncio.run(agent.propose_trade('tok', DummyPortfolio()))
    assert actions and actions[0]['side'] == 'buy'


async def fake_feed_low(token):
    return 1.0


async def fake_feed_high(token):
    return 1.2


def test_arbitrage_agent(monkeypatch):
    agent = ArbitrageAgent(threshold=0.1, amount=5, feeds=[fake_feed_low, fake_feed_high])
    actions = asyncio.run(agent.propose_trade('tok', DummyPortfolio()))
    assert {"side": "buy"} in [{"side": a['side']} for a in actions]
    assert {"side": "sell"} in [{"side": a['side']} for a in actions]


def test_exit_agent_trailing(monkeypatch):
    pf = DummyPortfolio()
    pf.balances['tok'] = Position('tok', 10, 2.0, 3.0)

    async def price_high(tokens):
        return {'tok': 2.5}

    monkeypatch.setattr(
        'solhunter_zero.agents.exit.fetch_token_prices_async',
        price_high,
    )

    agent = ExitAgent(trailing=0.2)
    actions = asyncio.run(agent.propose_trade('tok', pf))
    assert actions == []

    async def price_low(tokens):
        return {'tok': 2.0}

    monkeypatch.setattr(
        'solhunter_zero.agents.exit.fetch_token_prices_async',
        price_low,
    )
    actions = asyncio.run(agent.propose_trade('tok', pf))
    assert actions and actions[0]['side'] == 'sell'


def test_execution_agent(monkeypatch):
    captured = {}

    async def fake_place(token, side, amount, price, **_):
        captured['side'] = side
        return {'ok': True}

    monkeypatch.setattr('solhunter_zero.agents.execution.place_order_async', fake_place)
    agent = ExecutionAgent(rate_limit=0)
    res = asyncio.run(agent.execute({'token': 'tok', 'side': 'buy', 'amount': 1.0, 'price': 1.0}))
    assert captured['side'] == 'buy'
    assert res == {'ok': True}


def test_agent_manager_execute(monkeypatch):
    async def buy_agent(token, portfolio):
        return [{'token': token, 'side': 'buy', 'amount': 1.0, 'price': 1.0}]

    class DummyAgent:
        async def propose_trade(self, token, portfolio):
            return [{'token': token, 'side': 'sell', 'amount': 1.0, 'price': 1.5}]

    captured = []

    async def fake_place(token, side, amount, price, **_):
        captured.append((side, amount))
        return {'ok': True}

    monkeypatch.setattr('solhunter_zero.agents.execution.place_order_async', fake_place)
    exec_agent = ExecutionAgent(rate_limit=0)
    mgr = AgentManager([types.SimpleNamespace(propose_trade=buy_agent), DummyAgent()], executor=exec_agent)

    pf = DummyPortfolio()
    asyncio.run(mgr.execute('tok', pf))
    assert ('buy', 1.0) in captured and ('sell', 1.0) in captured


def test_memory_agent(monkeypatch):
    mem_agent = MemoryAgent()
    asyncio.run(mem_agent.log({'token': 'tok', 'side': 'buy', 'amount': 1.0, 'price': 2.0}))
    trades = mem_agent.memory.list_trades()
    assert trades and trades[0].token == 'tok'


def test_agent_manager_logs_trades(monkeypatch):
    async def propose(token, portfolio):
        return [{'token': token, 'side': 'buy', 'amount': 1.0, 'price': 1.0}]

    memory_agent = MemoryAgent()
    log_called = {}

    async def fake_log(action, *, skip_db=False):
        log_called['args'] = (action, skip_db)

    monkeypatch.setattr(memory_agent, 'log', fake_log)

    exec_agent = ExecutionAgent(rate_limit=0, dry_run=True)

    async def fake_execute(action):
        return {'ok': True}

    monkeypatch.setattr(exec_agent, 'execute', fake_execute)

    mgr = AgentManager([types.SimpleNamespace(propose_trade=propose), memory_agent], executor=exec_agent)
    pf = DummyPortfolio()
    asyncio.run(mgr.execute('tok', pf))

    assert log_called['args'][0]['side'] == 'buy'
    assert log_called['args'][1] is True

