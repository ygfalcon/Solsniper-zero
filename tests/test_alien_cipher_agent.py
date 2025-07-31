import asyncio

from solhunter_zero.agents.alien_cipher_agent import AlienCipherAgent
from solhunter_zero.portfolio import Portfolio, Position


class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}


def test_alien_cipher_buy():
    data = {"tok": [{"x_n": 0.9}]}
    agent = AlienCipherAgent(data=data, threshold=0.8)
    pf = DummyPortfolio()
    actions = asyncio.run(agent.propose_trade("tok", pf))
    assert actions and actions[0]["side"] == "buy"


def test_alien_cipher_sell():
    data = {"tok": [{"x_n": 0.1}]}
    agent = AlienCipherAgent(data=data, threshold=0.7)
    pf = DummyPortfolio()
    pf.balances["tok"] = Position("tok", 1.0, 1.0)
    actions = asyncio.run(agent.propose_trade("tok", pf))
    assert actions and actions[0]["side"] == "sell"
