from __future__ import annotations

import asyncio
from typing import Iterable, Dict, Any, List

from .agents import BaseAgent
from .agents.execution import ExecutionAgent
from .agents.swarm import AgentSwarm
from .agents.memory import MemoryAgent



class AgentManager:
    """Manage and coordinate trading agents and execute actions."""

    def __init__(
        self,
        agents: Iterable[BaseAgent],
        executor: ExecutionAgent | None = None,
        *,
        weights: Dict[str, float] | None = None,
        memory_agent: MemoryAgent | None = None,
    ):
        self.agents = list(agents)
        self.executor = executor or ExecutionAgent()
        self.weights = weights or {}
        self.memory_agent = memory_agent or next(
            (a for a in self.agents if isinstance(a, MemoryAgent)),
            None,
        )

    async def evaluate(self, token: str, portfolio) -> List[Dict[str, Any]]:
        swarm = AgentSwarm(self.agents)
        return await swarm.propose(token, portfolio, weights=self.weights)


    async def execute(self, token: str, portfolio) -> List[Any]:
        actions = await self.evaluate(token, portfolio)
        results = []
        for action in actions:
            results.append(await self.executor.execute(action))
            if self.memory_agent:
                await self.memory_agent.log(action)
        return results

    def update_weights(self) -> None:
        """Adjust agent weights based on historical trade ROI."""
        if not self.memory_agent:
            return

        trades = self.memory_agent.memory.list_trades()
        summary: Dict[str, Dict[str, float]] = {}
        for t in trades:
            name = t.reason or ""
            info = summary.setdefault(name, {"buy": 0.0, "sell": 0.0})
            info[t.direction] += t.amount * t.price

        for name, info in summary.items():
            spent = info.get("buy", 0.0)
            revenue = info.get("sell", 0.0)
            if spent <= 0:
                continue
            roi = (revenue - spent) / spent
            if roi > 0:
                self.weights[name] = self.weights.get(name, 1.0) * 1.1
            elif roi < 0:
                self.weights[name] = self.weights.get(name, 1.0) * 0.9

