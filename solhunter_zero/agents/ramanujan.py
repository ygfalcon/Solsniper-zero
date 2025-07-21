from __future__ import annotations

from typing import List, Dict, Any

from . import BaseAgent
from ..portfolio import Portfolio


class RamanujanAgent(BaseAgent):
    """Toy agent using a simple numeric heuristic."""

    name = "ramanujan"

    def __init__(self, threshold: int = 1729) -> None:
        self.threshold = threshold

    @staticmethod
    def _score(token: str) -> int:
        return sum(ord(c) for c in token)

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        score = self._score(token)
        if score > self.threshold:
            return [{"token": token, "side": "buy", "amount": 1.0, "price": 0.0}]
        pos = portfolio.balances.get(token)
        if score < self.threshold and pos:
            return [{"token": token, "side": "sell", "amount": pos.amount, "price": 0.0}]
        return []
