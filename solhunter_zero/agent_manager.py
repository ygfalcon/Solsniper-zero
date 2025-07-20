from __future__ import annotations

import asyncio
from typing import Iterable, Dict, Any, List

from .agents import BaseAgent
from .agents.execution import ExecutionAgent
from .agents.memory import MemoryAgent


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
        mem_agent = next(
            (a for a in self.agents if isinstance(a, MemoryAgent)), None
        )
        for action in actions:
            res = await self.executor.execute(action)
            results.append(res)
            if mem_agent:
                await mem_agent.log(action, skip_db=self.executor.dry_run)
        return results
