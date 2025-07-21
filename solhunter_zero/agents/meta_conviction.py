from __future__ import annotations

import asyncio
from typing import List, Dict, Any

from . import BaseAgent
from .simulation import SimulationAgent
from .conviction import ConvictionAgent
from .ramanujan_agent import RamanujanAgent
from ..portfolio import Portfolio


class MetaConvictionAgent(BaseAgent):
    """Combine several conviction agents into a single decision."""

    name = "meta_conviction"

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.sim_agent = SimulationAgent()
        self.conv_agent = ConvictionAgent()
        self.ram_agent = RamanujanAgent()
        self.weights = weights or {
            "simulation": 1.0,
            "conviction": 1.0,
            "ramanujan": 1.0,
        }

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        results = await asyncio.gather(
            self.sim_agent.propose_trade(token, portfolio),
            self.conv_agent.propose_trade(token, portfolio),
            self.ram_agent.propose_trade(token, portfolio),
        )

        agents = [self.sim_agent, self.conv_agent, self.ram_agent]
        conviction = 0.0
        for agent, res in zip(agents, results):
            if not res:
                continue
            side = res[0].get("side")
            weight = self.weights.get(agent.name, 1.0)
            if side == "buy":
                conviction += weight
            elif side == "sell":
                conviction -= weight

        if conviction > 0:
            return [{"token": token, "side": "buy", "amount": 1.0, "price": 0.0}]
        if conviction < 0:
            pos = portfolio.balances.get(token)
            if pos:
                return [{"token": token, "side": "sell", "amount": pos.amount, "price": 0.0}]
        return []
