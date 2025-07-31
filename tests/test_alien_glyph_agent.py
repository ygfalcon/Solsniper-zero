import asyncio

from solhunter_zero.agents.alien_glyph_agent import AlienGlyphAgent
from solhunter_zero.portfolio import Portfolio, Position


class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}


def test_alien_glyph_agent_buy():
    loader = lambda token: [1, 2, 3]
    agent = AlienGlyphAgent(threshold=1.0, glyph_loader=loader)
    pf = DummyPortfolio()
    actions = asyncio.run(agent.propose_trade("tok", pf))
    assert actions and actions[0]["side"] == "buy"


def test_alien_glyph_agent_sell():
    loader = lambda token: [-3, -2, -4]
    agent = AlienGlyphAgent(threshold=1.0, glyph_loader=loader)
    pf = DummyPortfolio()
    pf.balances["tok"] = Position("tok", 2.0, 1.0)
    actions = asyncio.run(agent.propose_trade("tok", pf))
    assert actions and actions[0]["side"] == "sell"
