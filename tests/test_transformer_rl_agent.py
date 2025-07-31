import asyncio

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from solhunter_zero.agents.transformer_rl import TransformerRLAgent
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.memory import Memory
from solhunter_zero.portfolio import Portfolio, Position

class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}

def setup_memory():
    mem = Memory("sqlite:///:memory:")
    loop.run_until_complete(mem.log_trade(token="tok", direction="buy", amount=1, price=1))
    loop.run_until_complete(mem.log_trade(token="tok", direction="sell", amount=1, price=2))
    trades = loop.run_until_complete(mem.list_trades())
    mem.list_trades = lambda since_id=None, **_: trades
    return mem

def test_transformer_agent_buy(tmp_path):
    mem = setup_memory()
    from solhunter_zero import order_book_ws
    order_book_ws.snapshot = lambda *a, **k: (0.0, 0.0, 0.0)
    mem_agent = MemoryAgent(mem)
    agent = TransformerRLAgent(memory_agent=mem_agent, model_path=tmp_path / "tf.pt")
    pf = DummyPortfolio()
    actions = loop.run_until_complete(agent.propose_trade("tok", pf))
    assert isinstance(actions, list)

def test_transformer_agent_sell(tmp_path):
    mem = setup_memory()
    from solhunter_zero import order_book_ws
    order_book_ws.snapshot = lambda *a, **k: (0.0, 0.0, 0.0)
    mem_agent = MemoryAgent(mem)
    agent = TransformerRLAgent(memory_agent=mem_agent, model_path=tmp_path / "tf.pt")
    pf = DummyPortfolio()
    pf.balances["tok"] = Position("tok", 1, 1.0, 1.0)
    actions = loop.run_until_complete(agent.propose_trade("tok", pf))
    assert actions
    assert agent._fitted
