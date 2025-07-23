from __future__ import annotations

import asyncio
from typing import List, Dict, Any

from . import BaseAgent
from .portfolio_optimizer import PortfolioOptimizer
from .execution import ExecutionAgent
from ..depth_client import snapshot
from ..arbitrage import _prepare_service_tx, VENUE_URLS, DEX_BASE_URL
from ..mev_executor import MEVExecutor
from ..portfolio import Portfolio


class CrossDEXRebalancer(BaseAgent):
    """Rebalance trades across DEX venues based on order book liquidity."""

    name = "crossdex_rebalancer"

    def __init__(
        self,
        optimizer: PortfolioOptimizer | None = None,
        executor: ExecutionAgent | None = None,
        *,
        rebalance_interval: int = 30,
        slippage_threshold: float = 0.05,
        use_mev_bundles: bool = False,
    ) -> None:
        self.optimizer = optimizer or PortfolioOptimizer()
        self.executor = executor or ExecutionAgent(rate_limit=0)
        self.rebalance_interval = int(rebalance_interval)
        self.slippage_threshold = float(slippage_threshold)
        self.use_mev_bundles = bool(use_mev_bundles)
        self._last = 0.0

    # ------------------------------------------------------------------
    async def _split_action(
        self, action: Dict[str, Any], depth: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        side = action.get("side", "buy").lower()
        amount = float(action.get("amount", 0.0))
        if amount <= 0:
            return []

        slip: Dict[str, float] = {}
        key = "asks" if side == "buy" else "bids"
        for venue, info in depth.items():
            liq = float(info.get(key, 0.0))
            if liq > 0:
                slip[venue] = amount / liq
        if not slip:
            return [action]

        valid = {v: s for v, s in slip.items() if s <= self.slippage_threshold}
        if not valid:
            best = min(slip, key=slip.get)
            valid = {best: slip[best]}

        inv = {v: 1.0 / max(s, 1e-9) for v, s in valid.items()}
        total = sum(inv.values())
        actions: List[Dict[str, Any]] = []
        for venue, inv_w in inv.items():
            amt = amount * inv_w / total
            new = dict(action)
            new["venue"] = venue
            new["amount"] = amt
            actions.append(new)
        return actions

    # ------------------------------------------------------------------
    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        now = asyncio.get_event_loop().time()
        if now - self._last < self.rebalance_interval:
            return []
        self._last = now

        base_actions = await self.optimizer.propose_trade(
            token, portfolio, depth=depth, imbalance=imbalance
        )
        if not base_actions:
            return []

        depth_data, _ = snapshot(token)
        all_actions: List[Dict[str, Any]] = []
        for act in base_actions:
            all_actions.extend(await self._split_action(act, depth_data))

        if self.use_mev_bundles:
            txs: List[str] = []
            for act in all_actions:
                venue = str(act.get("venue", ""))
                base = VENUE_URLS.get(venue, DEX_BASE_URL)
                tx = await _prepare_service_tx(
                    act["token"],
                    act["side"],
                    act.get("amount", 0.0),
                    act.get("price", 0.0),
                    base,
                )
                if tx:
                    txs.append(tx)
            if txs:
                mev = MEVExecutor(
                    token,
                    priority_rpc=getattr(self.executor, "priority_rpc", None),
                )
                await mev.submit_bundle(txs)
                return [{"bundle": True, "count": len(txs)}]
            return []

        results = []
        for act in all_actions:
            res = await self.executor.execute(act)
            results.append(res)
        return results
