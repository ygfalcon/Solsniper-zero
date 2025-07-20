from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Sequence, Callable, Awaitable

from . import BaseAgent
from .. import arbitrage
from ..portfolio import Portfolio

PriceFeed = Callable[[str], Awaitable[float]]


class ArbitrageAgent(BaseAgent):
    """Detect arbitrage opportunities between DEX price feeds."""

    name = "arbitrage"

    def __init__(self, threshold: float = 0.0, amount: float = 1.0, feeds: Sequence[PriceFeed] | None = None):
        self.threshold = threshold
        self.amount = amount
        self.feeds = feeds or [arbitrage.fetch_orca_price_async, arbitrage.fetch_raydium_price_async]

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        prices = await asyncio.gather(*(feed(token) for feed in self.feeds))
        if not prices:
            return []
        min_price = min(prices)
        max_price = max(prices)
        if min_price <= 0:
            return []
        diff = (max_price - min_price) / min_price
        if diff < self.threshold:
            return []
        return [
            {"token": token, "side": "buy", "amount": self.amount, "price": min_price},
            {"token": token, "side": "sell", "amount": self.amount, "price": max_price},
        ]
