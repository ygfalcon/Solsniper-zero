from solhunter_zero.rl_training import MultiAgentRL
from solhunter_zero.agents.hierarchical_rl_agent import HierarchicalRLAgent
from solhunter_zero.swarm_coordinator import SwarmCoordinator
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.advanced_memory import AdvancedMemory


class DummyAgent:
    def __init__(self, name):
        self.name = name

    async def propose_trade(self, token, portfolio, *, depth=None, imbalance=None):
        return []


def test_hierarchical_rl_agent_trains(tmp_path):
    rl = MultiAgentRL(population_size=2, model_base=str(tmp_path / "m.pt"))
    rl.controller.population = [
        {"weights": {"a1": 0.5, "a2": 1.5}, "risk": {"risk_multiplier": 1.0}},
        {"weights": {"a1": 1.0, "a2": 1.0}, "risk": {"risk_multiplier": 1.0}},
    ]
    agent = HierarchicalRLAgent(rl)
    w = agent.train(["a1", "a2"])
    assert w["a1"] == 0.5
    assert w["a2"] == 1.5


def test_swarm_coordinator_filters_hierarchical_agent(tmp_path):
    mem = AdvancedMemory(url="sqlite:///:memory:")
    mem_agent = MemoryAgent(mem)
    rl = MultiAgentRL(population_size=2, model_base=str(tmp_path / "m.pt"))
    rl.controller.population = [
        {"weights": {"a1": 0.5, "a2": 1.5}, "risk": {"risk_multiplier": 1.0}},
        {"weights": {"a1": 1.0, "a2": 1.0}, "risk": {"risk_multiplier": 1.0}},
    ]
    hier = HierarchicalRLAgent(rl)
    agents = [DummyAgent("a1"), DummyAgent("a2"), hier]
    coord = SwarmCoordinator(mem_agent, {"a1": 1.0, "a2": 1.0})
    w = coord.compute_weights(agents)
    assert w["a1"] < w["a2"]
