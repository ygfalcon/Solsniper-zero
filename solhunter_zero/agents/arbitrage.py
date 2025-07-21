from __future__ import annotations

import asyncio
import os
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
        backup_feeds: Mapping[str, PriceFeed] | Sequence[PriceFeed] | None = None,
        *,
        fees: Mapping[str, float] | None = None,
        gas: Mapping[str, float] | None = None,
        latency: Mapping[str, float] | None = None,
        gas_multiplier: float | None = None,
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
        self.fees = dict(fees or {})
        self.gas = dict(gas or {})
        self.latency = dict(latency or {})
        if gas_multiplier is not None:
            self.gas_multiplier = float(gas_multiplier)
        else:
            self.gas_multiplier = float(os.getenv("GAS_MULTIPLIER", "1.0"))
        if backup_feeds is None:
            self.backup_feeds: Dict[str, PriceFeed] | None = None
        elif isinstance(backup_feeds, Mapping):
            self.backup_feeds = dict(backup_feeds)
        else:
            self.backup_feeds = {
                getattr(f, "__name__", f"backup{i}"): f
                for i, f in enumerate(backup_feeds)
            }
        # Cache of latest known prices per token and feed
        self.price_cache: Dict[str, Dict[str, float]] = {}

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        token_cache = self.price_cache.setdefault(token, {})

        # Fetch prices from main feeds
        names = list(self.feeds.keys())
        prices = await asyncio.gather(*(f(token) for f in self.feeds.values()))

        valid: Dict[str, float] = {}
        for name, price in zip(names, prices):
            if price > 0:
                token_cache[name] = price
                valid[name] = price
            elif name in token_cache:
                valid[name] = token_cache[name]

        # If all feeds failed, try backup feeds
        if len(valid) < 2 and self.backup_feeds:
            b_names = list(self.backup_feeds.keys())
            b_prices = await asyncio.gather(
                *(f(token) for f in self.backup_feeds.values())
            )
            for b_name, b_price in zip(b_names, b_prices):
                if b_price > 0:
                    token_cache[b_name] = b_price
                    valid[b_name] = b_price
                elif b_name in token_cache:
                    valid[b_name] = token_cache[b_name]

        # If still not enough data, give up
        if len(valid) < 2:
            return []

        names = list(valid.keys())
        prices = list(valid.values())
        min_price = min(prices)
        max_price = max(prices)
        if min_price <= 0:
            return []
        diff = (max_price - min_price) / min_price
        if diff < self.threshold:
            return []

        buy_idx = prices.index(min_price)
        sell_idx = prices.index(max_price)
        buy_name = names[buy_idx]
        sell_name = names[sell_idx]

        fee_cost = (
            min_price * self.amount * self.fees.get(buy_name, 0.0)
            + max_price * self.amount * self.fees.get(sell_name, 0.0)
        )
        gas_cost = (
            self.gas.get(buy_name, 0.0) + self.gas.get(sell_name, 0.0)
        ) * self.gas_multiplier
        latency_cost = self.latency.get(buy_name, 0.0) + self.latency.get(sell_name, 0.0)

        profit = (max_price - min_price) * self.amount - fee_cost - gas_cost - latency_cost

        if profit <= 0:
            return []

        return [
            {
                "token": token,
                "side": "buy",
                "amount": self.amount,
                "price": min_price,
                "venue": buy_name,
            },
            {
                "token": token,
                "side": "sell",
                "amount": self.amount,
                "price": max_price,
                "venue": sell_name,
            },
        ]
