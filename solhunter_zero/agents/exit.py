from __future__ import annotations

from typing import List, Dict, Any

from . import BaseAgent
from ..portfolio import Portfolio
from ..prices import fetch_token_prices_async


class ExitAgent(BaseAgent):
    """Propose exit trades using trailing stops."""

    name = "exit"

    def __init__(self, trailing: float = 0.0):
        self.trailing = trailing

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        if token not in portfolio.balances:
            return []
        prices = await fetch_token_prices_async({token})
        price = prices.get(token, 0.0)
        if self.trailing and portfolio.trailing_stop_triggered(token, price, self.trailing):
            pos = portfolio.balances[token]
            return [{"token": token, "side": "sell", "amount": pos.amount, "price": price}]
        return []
