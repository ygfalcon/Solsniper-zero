from __future__ import annotations

import asyncio
from typing import Iterable, Dict, Any, List

from .agents import BaseAgent
from .agents.execution import ExecutionAgent
from .agents.swarm import AgentSwarm


class AgentManager:
    """Manage and coordinate trading agents and execute actions."""

    def __init__(
        self,
        agents: Iterable[BaseAgent],
        executor: ExecutionAgent | None = None,
        *,
        weights: Dict[str, float] | None = None,
    ):
        self.agents = list(agents)
        self.executor = executor or ExecutionAgent()
        self.weights = weights or {}

    async def evaluate(self, token: str, portfolio) -> List[Dict[str, Any]]:
        swarm = AgentSwarm(self.agents)
        return await swarm.propose(token, portfolio, weights=self.weights)

    async def execute(self, token: str, portfolio) -> List[Any]:
        actions = await self.evaluate(token, portfolio)
        results = []
        for action in actions:
            results.append(await self.executor.execute(action))
        return results
