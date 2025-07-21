from __future__ import annotations

from typing import List, Dict, Any

from . import BaseAgent
from ..portfolio import Portfolio


class PortfolioAgent(BaseAgent):
    """Agent that enforces portfolio allocation limits."""

    name = "portfolio"

    def __init__(self, max_allocation: float = 0.2, amount: float = 1.0) -> None:
        self.max_allocation = max_allocation
        self.amount = amount

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        alloc = portfolio.percent_allocated(token)
        if alloc > self.max_allocation:
            pos = portfolio.balances.get(token)
            if pos:
                return [{"token": token, "side": "sell", "amount": pos.amount, "price": 0.0}]
            return []

        if alloc < self.max_allocation and portfolio.total_value({}) > 0:
            return [{"token": token, "side": "buy", "amount": self.amount, "price": 0.0}]

        return []
