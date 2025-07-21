import asyncio
import types

import pytest

from solhunter_zero.agents.simulation import SimulationAgent
from solhunter_zero.agents.conviction import ConvictionAgent
from solhunter_zero.agents.arbitrage import ArbitrageAgent
from solhunter_zero.agents.exit import ExitAgent
from solhunter_zero.agents.execution import ExecutionAgent
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.agents.swarm import AgentSwarm
from solhunter_zero.memory import Memory

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


def test_exit_agent_stop_loss(monkeypatch):
    pf = DummyPortfolio()
    pf.balances['tok'] = Position('tok', 10, 10.0, 10.0)

    async def price_low(tokens):
        return {'tok': 8.0}

    monkeypatch.setattr(
        'solhunter_zero.agents.exit.fetch_token_prices_async',
        price_low,
    )

    agent = ExitAgent(stop_loss=0.2)
    actions = asyncio.run(agent.propose_trade('tok', pf))
    assert actions and actions[0]['side'] == 'sell'


def test_exit_agent_take_profit(monkeypatch):
    pf = DummyPortfolio()
    pf.balances['tok'] = Position('tok', 5, 10.0, 10.0)

    async def price_high(tokens):
        return {'tok': 12.0}

    monkeypatch.setattr(
        'solhunter_zero.agents.exit.fetch_token_prices_async',
        price_high,
    )

    agent = ExitAgent(take_profit=0.2)
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
        name = 'dummy'
        async def propose_trade(self, token, portfolio):
            return [{'token': token, 'side': 'sell', 'amount': 1.0, 'price': 1.5}]

    captured = []

    async def fake_place(token, side, amount, price, **_):
        captured.append((side, amount))
        return {'ok': True}

    monkeypatch.setattr('solhunter_zero.agents.execution.place_order_async', fake_place)
    exec_agent = ExecutionAgent(rate_limit=0)
    mgr = AgentManager(
        [types.SimpleNamespace(propose_trade=buy_agent, name='b'), DummyAgent()],
        executor=exec_agent,
        memory_agent=None,
    )

    pf = DummyPortfolio()
    asyncio.run(mgr.execute('tok', pf))
    # Buy and sell of equal size cancel out -> no orders executed
    assert captured == []


def test_agent_swarm_weighted(monkeypatch):
    async def buy_one(token, pf):
        return [{'token': token, 'side': 'buy', 'amount': 1.0, 'price': 1.0}]

    async def buy_two(token, pf):
        return [{'token': token, 'side': 'buy', 'amount': 1.0, 'price': 2.0}]

    swarm = AgentSwarm([
        types.SimpleNamespace(propose_trade=buy_one, name='a'),
        types.SimpleNamespace(propose_trade=buy_two, name='b'),
    ])
    actions = asyncio.run(swarm.propose('tok', DummyPortfolio(), weights={'a': 1.0, 'b': 2.0}))
    assert actions == [{
        'token': 'tok',
        'side': 'buy',
        'amount': 3.0,
        'price': pytest.approx(5 / 3),
    }]


def test_agent_swarm_conflict_cancel():
    async def buy(token, pf):
        return [{'token': token, 'side': 'buy', 'amount': 1.0, 'price': 1.0}]

    async def sell(token, pf):
        return [{'token': token, 'side': 'sell', 'amount': 1.0, 'price': 1.5}]

    swarm = AgentSwarm([
        types.SimpleNamespace(propose_trade=buy, name='b'),
        types.SimpleNamespace(propose_trade=sell, name='s'),
    ])
    actions = asyncio.run(swarm.propose('tok', DummyPortfolio()))
    assert actions == []


def test_memory_agent(monkeypatch):
    mem_agent = MemoryAgent()
    asyncio.run(mem_agent.log({'token': 'tok', 'side': 'buy', 'amount': 1.0, 'price': 2.0}))
    trades = mem_agent.memory.list_trades()
    assert trades and trades[0].token == 'tok'


def test_agent_manager_update_weights():
    mem = Memory('sqlite:///:memory:')
    mem_agent = MemoryAgent(mem)
    mgr = AgentManager([], memory_agent=mem_agent, weights={'a1': 1.0, 'a2': 1.0})

    mem.log_trade(token='tok', direction='buy', amount=1, price=1, reason='a1')
    mem.log_trade(token='tok', direction='sell', amount=1, price=2, reason='a1')
    mem.log_trade(token='tok', direction='buy', amount=1, price=2, reason='a2')
    mem.log_trade(token='tok', direction='sell', amount=1, price=1, reason='a2')

    mgr.update_weights()

    assert mgr.weights['a1'] > 1.0
    assert mgr.weights['a2'] < 1.0


def test_execution_agent_parallel(monkeypatch):
    async def fake_place(*a, **k):
        await asyncio.sleep(0.1)
        return {'ok': True}

    monkeypatch.setattr(
        'solhunter_zero.agents.execution.place_order_async', fake_place
    )
    agent = ExecutionAgent(rate_limit=0, concurrency=2)

    async def run_two():
        start = asyncio.get_event_loop().time()
        await asyncio.gather(
            agent.execute({'token': 'tok', 'side': 'buy', 'amount': 1.0, 'price': 1.0}),
            agent.execute({'token': 'tok', 'side': 'sell', 'amount': 1.0, 'price': 1.0}),
        )
        return asyncio.get_event_loop().time() - start

    elapsed = asyncio.run(run_two())
    assert elapsed < 0.2


def test_execution_agent_rate_limit(monkeypatch):
    call_times = []

    async def fake_place(*a, **k):
        call_times.append(asyncio.get_event_loop().time())
        return {'ok': True}

    monkeypatch.setattr(
        'solhunter_zero.agents.execution.place_order_async', fake_place
    )
    agent = ExecutionAgent(rate_limit=0.05, concurrency=2)

    async def run_two():
        await asyncio.gather(
            agent.execute({'token': 'tok', 'side': 'buy', 'amount': 1.0, 'price': 1.0}),
            agent.execute({'token': 'tok', 'side': 'buy', 'amount': 1.0, 'price': 1.0}),
        )

    asyncio.run(run_two())
    assert len(call_times) == 2
    assert call_times[1] - call_times[0] >= 0.05

