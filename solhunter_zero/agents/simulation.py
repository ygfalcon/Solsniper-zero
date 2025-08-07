from __future__ import annotations

from typing import Any, Dict, List

from ..decision import should_buy, should_sell
from ..portfolio import Portfolio
from ..prices import fetch_token_prices_async
from ..simulation import run_simulations
from . import BaseAgent


class SimulationAgent(BaseAgent):
    """Run Monte Carlo simulations and propose trades based on the results."""

    name = "simulation"

    def __init__(self, count: int = 100):
        self.count = count

    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        sims = run_simulations(token, count=self.count)
        if not sims:
            return []

        prices = await fetch_token_prices_async({token})
        price = prices.get(token, 0.0)

        actions: List[Dict[str, Any]] = []
        if should_sell(sims):
            pos = portfolio.balances.get(token)
            if pos:
                actions.append(
                    {
                        "token": token,
                        "side": "sell",
                        "amount": pos.amount,
                        "price": price,
                    }
                )
        elif should_buy(sims):
            actions.append(
                {"token": token, "side": "buy", "amount": 1.0, "price": price}
            )
        return actions
