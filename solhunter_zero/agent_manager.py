from __future__ import annotations

import asyncio
from typing import Iterable, Dict, Any, List

from .agents import BaseAgent
from .agents.execution import ExecutionAgent
from .agents.simulation import SimulationAgent
from .agents.conviction import ConvictionAgent
from .agents.arbitrage import ArbitrageAgent
from .agents.exit import ExitAgent
from .agents.memory import MemoryAgent
from .agents.discovery import DiscoveryAgent

_AGENT_MAP = {
    "simulation": SimulationAgent,
    "conviction": ConvictionAgent,
    "arbitrage": ArbitrageAgent,
    "exit": ExitAgent,
    "memory": MemoryAgent,
    "discovery": DiscoveryAgent,
}


def load_agent(name: str, *, weight: float | None = None) -> BaseAgent:
    """Return a new agent instance for ``name``.

    Parameters
    ----------
    name:
        Short agent name. Must be one of the built-in agent names.
    weight:
        Optional weight applied to proposed trade amounts.
    """

    cls = _AGENT_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown agent: {name}")
    agent = cls()
    if weight is not None:
        setattr(agent, "weight", float(weight))
    return agent


class AgentManager:
    """Manage and coordinate trading agents and execute actions."""

    def __init__(self, agents: Iterable[BaseAgent], executor: ExecutionAgent | None = None):
        self.agents = list(agents)
        self.executor = executor or ExecutionAgent()

    # ------------------------------------------------------------------
    #  Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "AgentManager":
        """Create ``AgentManager`` from configuration dictionary."""

        agent_names = cfg.get("agents", [])
        weights = cfg.get("agent_weights", {}) or {}

        agents: list[BaseAgent] = []
        for name in agent_names:
            weight = weights.get(name)
            try:
                agent = load_agent(name, weight=weight)
            except Exception:
                continue
            agents.append(agent)

        executor = ExecutionAgent()
        return cls(agents, executor=executor)

    async def evaluate(self, token: str, portfolio) -> List[Dict[str, Any]]:
        async def run(agent: BaseAgent):
            return await agent.propose_trade(token, portfolio)

        results = await asyncio.gather(*(run(a) for a in self.agents))
        actions: List[Dict[str, Any]] = []
        for res in results:
            if res:
                actions.extend(res)
        return actions

    async def execute(self, token: str, portfolio) -> List[Any]:
        actions = await self.evaluate(token, portfolio)
        results = []
        for action in actions:
            results.append(await self.executor.execute(action))
        return results
