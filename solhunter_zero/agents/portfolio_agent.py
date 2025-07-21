from __future__ import annotations

from typing import List, Dict, Any

from . import BaseAgent
from ..portfolio import Portfolio
from ..prices import fetch_token_prices_async


class PortfolioAgent(BaseAgent):
    """Enforce maximum portfolio exposure per token."""

    name = "portfolio"

    def __init__(self, max_exposure: float = 0.5) -> None:
        self.max_exposure = float(max_exposure)

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        if token not in portfolio.balances:
            return []

        # Fetch prices for all holdings to evaluate allocation
        prices = await fetch_token_prices_async(portfolio.balances.keys())
        if not prices:
            prices = {t: p.entry_price for t, p in portfolio.balances.items()}

        total = portfolio.total_value(prices)
        if total <= 0:
            return []

        pos = portfolio.balances[token]
        price = prices.get(token, pos.entry_price)
        exposure = (pos.amount * price) / total
        if exposure <= self.max_exposure:
            return []

        target_value = total * self.max_exposure
        target_amount = target_value / price
        sell_amount = max(0.0, pos.amount - target_amount)
        if sell_amount <= 0:
            return []

        return [{"token": token, "side": "sell", "amount": sell_amount, "price": price}]
