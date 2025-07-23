import asyncio

from solhunter_zero.agents.cardinal_agent import CardinalAgent
from solhunter_zero.agents.swarm import AgentSwarm
from solhunter_zero.portfolio import Portfolio


class DummyMemory:
    def simulation_success_rate(self, token):
        return 0.0

    def log_simulation(self, token, expected_roi, success_prob):
        pass


class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}


def test_cardinal_agent_inversion():
    agent = CardinalAgent(threshold=0.6)
    swarm = AgentSwarm([agent], memory=DummyMemory())
    pf = DummyPortfolio()

    actions1 = asyncio.run(swarm.propose("TOK", pf))
    assert agent.symbolic_execution_stack == [1, 2, 3]
    swarm.record_results([{"ok": False} for _ in actions1])

    actions2 = asyncio.run(swarm.propose("TOK", pf))
    assert agent.inversion_gate_triggered
    assert agent.symbolic_execution_stack == [3, 1, 2]
    assert actions2

