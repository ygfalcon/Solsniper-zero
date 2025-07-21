import types
from solhunter_zero.agent_manager import StrategySelector
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.memory import Memory

class DummyAgent:
    def __init__(self, name):
        self.name = name


def test_strategy_selector_updates_ranking():
    mem = Memory('sqlite:///:memory:')
    mem_agent = MemoryAgent(mem)

    # initial trades
    mem.log_trade(token='tok', direction='buy', amount=1, price=1, reason='sniper')
    mem.log_trade(token='tok', direction='sell', amount=1, price=2, reason='sniper')

    mem.log_trade(token='tok', direction='buy', amount=1, price=1, reason='arbitrage')
    mem.log_trade(token='tok', direction='sell', amount=1, price=1.1, reason='arbitrage')

    mem.log_trade(token='tok', direction='buy', amount=1, price=1, reason='momentum')
    mem.log_trade(token='tok', direction='sell', amount=1, price=0.5, reason='momentum')

    selector = StrategySelector(mem_agent)
    agents = [DummyAgent('sniper'), DummyAgent('arbitrage'), DummyAgent('momentum')]
    ranking1 = selector.rank_agents(agents)
    assert ranking1[0] == 'sniper'

    # new profitable trades for arbitrage
    mem.log_trade(token='tok', direction='buy', amount=1, price=1, reason='arbitrage')
    mem.log_trade(token='tok', direction='sell', amount=1, price=3, reason='arbitrage')

    ranking2 = selector.rank_agents(agents)
    assert ranking2[0] == 'arbitrage'
