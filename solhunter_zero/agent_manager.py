from __future__ import annotations

import asyncio
from typing import Iterable, Dict, Any, List

from .agents import BaseAgent
from .agents.execution import ExecutionAgent
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
        self.weights: Dict[str, float] = {}
        for i, a in enumerate(self.agents):
            name = getattr(a, "name", f"agent{i}")
            self.weights.setdefault(name, 1.0)
        if weights:
            self.weights.update(weights)
        self.memory_agent = memory_agent

    async def evaluate(self, token: str, portfolio) -> List[Dict[str, Any]]:
        async def run(agent: BaseAgent):
            res = await agent.propose_trade(token, portfolio)
            if res:
                for r in res:
                    r.setdefault("agent", getattr(agent, "name", agent.__class__.__name__))
            return res

        results = await asyncio.gather(*(run(a) for a in self.agents))
        actions: List[Dict[str, Any]] = []
        for res in results:
            if not res:
                continue
            for r in res:
                weight = self.weights.get(r.get("agent", ""), 1.0)
                r = dict(r)
                r["amount"] = float(r.get("amount", 0.0)) * weight
                actions.append(r)
        return actions

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
