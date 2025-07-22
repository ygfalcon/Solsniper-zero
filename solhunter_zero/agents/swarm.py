from __future__ import annotations

import asyncio
from typing import Iterable, List, Dict, Any

from .. import order_book_ws

from . import BaseAgent
from ..portfolio import Portfolio
from ..advanced_memory import AdvancedMemory


class AgentSwarm:
    """Coordinate multiple agents and aggregate their proposals."""

    def __init__(self, agents: Iterable[BaseAgent] | None = None, *, memory: AdvancedMemory | None = None):
        self.agents: List[BaseAgent] = list(agents or [])
        self.memory = memory
        self._last_outcomes: Dict[str, bool | None] = {a.name: None for a in self.agents}
        self._last_actions: List[Dict[str, Any]] = []
        for a in self.agents:
            if memory is not None:
                setattr(a, "memory", memory)
            setattr(a, "swarm", self)
            setattr(a, "last_outcome", None)

    # ------------------------------------------------------------------
    def success_rate(self, token: str) -> float:
        """Return average recorded success probability for ``token``."""
        if not self.memory:
            return 0.0
        return self.memory.simulation_success_rate(token)

    # ------------------------------------------------------------------
    def record_results(self, results: List[Dict[str, Any]]) -> None:
        """Store execution results and update agent state."""
        if not self.memory:
            return
        by_agent: Dict[str, bool] = {}
        for action, res in zip(self._last_actions, results):
            name = action.get("agent")
            token = action.get("token")
            if not name or not token:
                continue
            ok = bool(res.get("ok", False))
            by_agent[name] = ok
            expected = float(action.get("expected_roi", 0.0))
            prob = 1.0 if ok else 0.0
            self.memory.log_simulation(token, expected_roi=expected, success_prob=prob)
        for agent in self.agents:
            if agent.name in by_agent:
                outcome = by_agent[agent.name]
                self._last_outcomes[agent.name] = outcome
                agent.last_outcome = outcome

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

        depth, imbalance, _ = order_book_ws.snapshot(token)

        async def run(agent: BaseAgent):
            if hasattr(agent, "last_outcome"):
                agent.last_outcome = self._last_outcomes.get(agent.name)
            return await agent.propose_trade(
                token,
                portfolio,
                depth=depth,
                imbalance=imbalance,
            )

        results = await asyncio.gather(*(run(a) for a in self.agents))

        weights_map = {a.name: float(weights.get(a.name, 1.0)) for a in self.agents} if weights else {}

        merged: Dict[tuple[str, str], Dict[str, Any]] = {}
        for agent, res in zip(self.agents, results):
            weight = weights_map.get(agent.name, 1.0)
            if not res:
                continue
            for r in res:
                r.setdefault("agent", agent.name)
                token = r.get("token")
                side = r.get("side")
                if not token or not side:
                    continue
                amt = float(r.get("amount", 0.0)) * weight
                price = float(r.get("price", 0.0))
                key = (token, side)
                m = merged.setdefault(key, {"token": token, "side": side, "amount": 0.0, "price": 0.0})
                for extra in ("conviction_delta", "regret", "misfires", "agent"):
                    if extra not in r:
                        continue
                    if extra == "agent":
                        if "agent" in m and m["agent"] != r["agent"]:
                            m.pop("agent", None)
                        elif "agent" not in m:
                            m["agent"] = r["agent"]
                    elif extra not in m:
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

        self._last_actions = list(final)
        return final
