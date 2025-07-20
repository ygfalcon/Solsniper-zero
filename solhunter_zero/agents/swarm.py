from __future__ import annotations

import asyncio
from typing import Iterable, List, Dict, Any

from . import BaseAgent
from ..portfolio import Portfolio


class AgentSwarm:
    """Coordinate multiple agents and aggregate their proposals."""

    def __init__(self, agents: Iterable[BaseAgent] | None = None):
        self.agents: List[BaseAgent] = list(agents or [])

    async def propose(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        async def run(agent: BaseAgent):
            return await agent.propose_trade(token, portfolio)

        results = await asyncio.gather(*(run(a) for a in self.agents))
        actions: List[Dict[str, Any]] = []
        for res in results:
            if res:
                actions.extend(res)
        return actions
