from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Sequence, Callable, Awaitable, Mapping

from . import BaseAgent
from .. import arbitrage
from ..portfolio import Portfolio

PriceFeed = Callable[[str], Awaitable[float]]


class ArbitrageAgent(BaseAgent):
    """Detect arbitrage opportunities between DEX price feeds."""

    name = "arbitrage"

    def __init__(
        self,
        threshold: float = 0.0,
        amount: float = 1.0,
        feeds: Mapping[str, PriceFeed] | Sequence[PriceFeed] | None = None,
    ):
        self.threshold = threshold
        self.amount = amount
        if feeds is None:
            self.feeds: Dict[str, PriceFeed] = {
                "orca": arbitrage.fetch_orca_price_async,
                "raydium": arbitrage.fetch_raydium_price_async,
            }
        elif isinstance(feeds, Mapping):
            self.feeds = dict(feeds)
        else:
            self.feeds = {
                getattr(f, "__name__", f"feed{i}"): f for i, f in enumerate(feeds)
            }

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        names = list(self.feeds.keys())
        prices = await asyncio.gather(*(f(token) for f in self.feeds.values()))
        if not prices:
            return []
        min_price = min(prices)
        max_price = max(prices)
        if min_price <= 0:
            return []
        diff = (max_price - min_price) / min_price
        if diff < self.threshold:
            return []
        buy_idx = prices.index(min_price)
        sell_idx = prices.index(max_price)
        return [
            {
                "token": token,
                "side": "buy",
                "amount": self.amount,
                "price": min_price,
                "venue": names[buy_idx],
            },
            {
                "token": token,
                "side": "sell",
                "amount": self.amount,
                "price": max_price,
                "venue": names[sell_idx],
            },
        ]
