from __future__ import annotations

import asyncio
from typing import Iterable, List, Dict, Any

from . import BaseAgent
from ..portfolio import Portfolio


class AgentSwarm:
    """Coordinate multiple agents and aggregate their proposals."""

    def __init__(self, agents: Iterable[BaseAgent] | None = None):
        self.agents: List[BaseAgent] = list(agents or [])

    async def propose(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        weights: Dict[str, float] | None = None,
    ) -> List[Dict[str, Any]]:
        """Gather proposals from all agents and return aggregated actions.

        Parameters
        ----------
        weights:
            Optional mapping of agent name to weighting factor applied to the
            amounts returned by that agent. Defaults to ``1.0`` for all agents.
        """

        async def run(agent: BaseAgent):
            return await agent.propose_trade(token, portfolio)

        results = await asyncio.gather(*(run(a) for a in self.agents))

        weights_map = {a.name: float(weights.get(a.name, 1.0)) for a in self.agents} if weights else {}

        merged: Dict[tuple[str, str], Dict[str, Any]] = {}
        for agent, res in zip(self.agents, results):
            weight = weights_map.get(agent.name, 1.0)
            if not res:
                continue
            for r in res:
                token = r.get("token")
                side = r.get("side")
                if not token or not side:
                    continue
                amt = float(r.get("amount", 0.0)) * weight
                price = float(r.get("price", 0.0))
                key = (token, side)
                m = merged.setdefault(key, {"token": token, "side": side, "amount": 0.0, "price": 0.0})
                for extra in ("conviction_delta", "regret", "misfires", "agent"):
                    if extra in r and extra not in m:
                        m[extra] = r[extra]
                old_amt = m["amount"]
                if old_amt + amt > 0:
                    m["price"] = (m["price"] * old_amt + price * amt) / (old_amt + amt)
                m["amount"] += amt

        final: List[Dict[str, Any]] = []
        tokens = {t for t, _ in merged.keys()}
        for tok in tokens:
            buy = merged.get((tok, "buy"), {"amount": 0.0, "price": 0.0})
            sell = merged.get((tok, "sell"), {"amount": 0.0, "price": 0.0})
            net = buy["amount"] - sell["amount"]
            if net > 0:
                entry = {"token": tok, "side": "buy", "amount": net, "price": buy["price"]}
                for extra in ("conviction_delta", "regret", "misfires", "agent"):
                    if extra in buy:
                        entry[extra] = buy[extra]
                final.append(entry)
            elif net < 0:
                entry = {"token": tok, "side": "sell", "amount": -net, "price": sell["price"]}
                for extra in ("conviction_delta", "regret", "misfires", "agent"):
                    if extra in sell:
                        entry[extra] = sell[extra]
                final.append(entry)

        return final
