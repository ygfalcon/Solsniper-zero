from __future__ import annotations

from typing import List, Dict, Any

from . import BaseAgent
from ..simulation import run_simulations, predict_price_movement
from ..portfolio import Portfolio


class ConvictionAgent(BaseAgent):
    """Basic conviction calculation using expected ROI."""

    name = "conviction"

    def __init__(self, threshold: float = 0.05, count: int = 100):
        self.threshold = threshold
        self.count = count

    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        sims = run_simulations(token, count=self.count, order_book_strength=depth)
        if not sims:
            return []
        avg_roi = sum(r.expected_roi for r in sims) / len(sims)
        try:
            pred = predict_price_movement(token)
        except Exception:
            pred = 0.0
        if abs(pred) >= self.threshold * 0.5:
            avg_roi = (avg_roi + pred) / 2
        if imbalance is not None:
            avg_roi += imbalance * 0.05
        if avg_roi > self.threshold:
            return [{"token": token, "side": "buy", "amount": 1.0, "price": 0.0}]
        if avg_roi < -self.threshold:
            pos = portfolio.balances.get(token)
            if pos:
                return [{"token": token, "side": "sell", "amount": pos.amount, "price": 0.0}]
        return []
