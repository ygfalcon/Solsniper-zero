from __future__ import annotations

import asyncio
from typing import Iterable, Dict, Any, List

from .agents import BaseAgent
from .agents.execution import ExecutionAgent


def _create_agent(name: str, cfg: dict) -> BaseAgent | None:
    """Return agent instance for ``name`` using ``cfg`` parameters."""
    name = name.lower()
    if name == "simulation":
        from .agents.simulation import SimulationAgent

        count = int(cfg.get("simulation_count", 100))
        return SimulationAgent(count=count)
    if name == "conviction":
        from .agents.conviction import ConvictionAgent

        threshold = float(cfg.get("conviction_threshold", 0.05))
        count = int(cfg.get("conviction_count", 100))
        return ConvictionAgent(threshold=threshold, count=count)
    if name == "arbitrage":
        from .agents.arbitrage import ArbitrageAgent

        threshold = float(cfg.get("arbitrage_threshold", 0.0))
        amount = float(cfg.get("arbitrage_amount", 1.0))
        return ArbitrageAgent(threshold=threshold, amount=amount)
    if name == "exit":
        from .agents.exit import ExitAgent

        trailing = float(cfg.get("trailing_stop", 0.0))
        return ExitAgent(trailing=trailing)
    if name == "memory":
        from .agents.memory import MemoryAgent

        return MemoryAgent()
    return None


class AgentManager:
    """Manage and coordinate trading agents and execute actions."""

    def __init__(self, agents: Iterable[BaseAgent], executor: ExecutionAgent | None = None):
        self.agents = list(agents)
        self.executor = executor or ExecutionAgent()

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

    @classmethod
    def from_config(cls, cfg: dict) -> "AgentManager":
        """Create ``AgentManager`` from configuration dictionary."""
        names = cfg.get("agents") or []
        if isinstance(names, str):
            names = [n.strip() for n in names.split(",") if n.strip()]

        agents: list[BaseAgent] = []
        for name in names:
            agent = _create_agent(str(name), cfg)
            if agent is not None:
                agents.append(agent)

        return cls(agents)
