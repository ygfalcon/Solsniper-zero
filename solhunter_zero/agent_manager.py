from __future__ import annotations

import asyncio
from typing import Iterable, Dict, Any, List

from .agents import BaseAgent
from .agents.execution import ExecutionAgent
from .agents.discovery import DiscoveryAgent
from .scanner import scan_tokens_async


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

    async def discover_tokens(
        self,
        *,
        offline: bool = False,
        token_file: str | None = None,
        method: str = "websocket",
    ) -> List[str]:
        """Return candidate tokens using the discovery agent if available."""

        for agent in self.agents:
            if isinstance(agent, DiscoveryAgent):
                try:
                    return await agent.discover_tokens(
                        offline=offline, token_file=token_file, method=method
                    )
                except TypeError:
                    return await agent.discover_tokens(
                        offline=offline, token_file=token_file
                    )

        try:
            return await scan_tokens_async(
                offline=offline, token_file=token_file, method=method
            )
        except TypeError:
            return await scan_tokens_async(
                offline=offline, token_file=token_file
            )

