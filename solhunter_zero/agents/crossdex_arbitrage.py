from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Mapping, Sequence, Callable, Awaitable

from . import BaseAgent
from ..arbitrage import DEX_FEES, DEX_GAS, DEX_LATENCY, VENUE_URLS, measure_dex_latency_async
from .. import routeffi as _routeffi
from ..portfolio import Portfolio

PriceFeed = Callable[[str], Awaitable[float]]


class CrossDEXArbitrage(BaseAgent):
    """Search multi-hop arbitrage paths across DEX venues."""

    name = "crossdex_arbitrage"

    def __init__(
        self,
        threshold: float = 0.0,
        amount: float = 1.0,
        feeds: Mapping[str, PriceFeed] | Sequence[PriceFeed] | None = None,
        *,
        max_hops: int = 4,
    ) -> None:
        self.threshold = float(threshold)
        self.amount = float(amount)
        if feeds is None:
            from .. import arbitrage as mod

            self.feeds: Dict[str, PriceFeed] = {
                "orca": mod.fetch_orca_price_async,
                "raydium": mod.fetch_raydium_price_async,
                "jupiter": mod.fetch_jupiter_price_async,
            }
        elif isinstance(feeds, Mapping):
            self.feeds = dict(feeds)
        else:
            self.feeds = {
                getattr(f, "__name__", f"feed{i}"): f for i, f in enumerate(feeds)
            }
        self.max_hops = int(max_hops)
        self._fees: Dict[str, float] = dict(DEX_FEES)
        self._gas: Dict[str, float] = dict(DEX_GAS)
        self._latency: Dict[str, float] = dict(DEX_LATENCY)
        self._latency_task: asyncio.Task | None = None
        self._latency_updates: Dict[str, float] = {}

    def close(self) -> None:
        if self._latency_task:
            self._latency_task.cancel()

    def _handle_latency(self, payload: Mapping[str, Any]) -> None:
        for venue, val in payload.items():
            try:
                self._latency_updates[venue] = float(val)
            except Exception:
                pass

    async def _ensure_latency(self) -> None:
        loop = asyncio.get_event_loop()
        if self._latency_updates:
            self._latency.update(self._latency_updates)
            self._latency_updates.clear()
        if self._latency_task is None:
            self._latency = await measure_dex_latency_async(VENUE_URLS)
            self._latency_task = loop.create_task(measure_dex_latency_async(VENUE_URLS))
        elif self._latency_task.done():
            try:
                res = self._latency_task.result()
                if isinstance(res, dict):
                    self._latency.update(res)
            except Exception:
                pass
            self._latency_task = loop.create_task(measure_dex_latency_async(VENUE_URLS))

    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        await self._ensure_latency()
        names = list(self.feeds.keys())
        prices = await asyncio.gather(*(f(token) for f in self.feeds.values()))
        price_map = {n: p for n, p in zip(names, prices) if p > 0}
        if len(price_map) < 2:
            return []

        func = _routeffi.best_route_parallel if _routeffi.parallel_enabled() else _routeffi.best_route
        res = func(
            price_map,
            self.amount,
            fees=self._fees,
            gas=self._gas,
            latency=self._latency,
            max_hops=self.max_hops,
        )
        if not res:
            return []
        path, profit = res
        if not path or profit <= 0:
            return []

        actions: List[Dict[str, Any]] = []
        for i in range(len(path) - 1):
            buy = path[i]
            sell = path[i + 1]
            actions.append({"token": token, "side": "buy", "amount": self.amount, "price": price_map[buy], "venue": buy})
            actions.append({"token": token, "side": "sell", "amount": self.amount, "price": price_map[sell], "venue": sell})
        return actions
