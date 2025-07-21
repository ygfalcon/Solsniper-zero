import asyncio

from solhunter_zero.agents.portfolio_agent import PortfolioAgent
from solhunter_zero.portfolio import Portfolio, Position


class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}


def test_portfolio_agent_sell(monkeypatch):
    pf = DummyPortfolio()
    pf.balances['tok'] = Position('tok', 8, 1.0, 1.0)
    pf.balances['oth'] = Position('oth', 2, 1.0, 1.0)

    async def fake_prices(tokens):
        return {'tok': 1.0, 'oth': 1.0}

    monkeypatch.setattr('solhunter_zero.agents.portfolio_agent.fetch_token_prices_async', fake_prices)

    agent = PortfolioAgent(max_exposure=0.5)
    actions = asyncio.run(agent.propose_trade('tok', pf))
    assert actions and actions[0]['side'] == 'sell'
    assert actions[0]['amount'] == 3.0


def test_portfolio_agent_no_action(monkeypatch):
    pf = DummyPortfolio()
    pf.balances['tok'] = Position('tok', 1, 1.0, 1.0)
    pf.balances['oth'] = Position('oth', 4, 1.0, 1.0)

    async def fake_prices(tokens):
        return {'tok': 1.0, 'oth': 1.0}

    monkeypatch.setattr('solhunter_zero.agents.portfolio_agent.fetch_token_prices_async', fake_prices)

    agent = PortfolioAgent(max_exposure=0.5)
    actions = asyncio.run(agent.propose_trade('tok', pf))
    assert actions == []
