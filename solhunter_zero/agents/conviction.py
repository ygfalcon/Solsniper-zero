from __future__ import annotations

from typing import List, Dict, Any

import os
import numpy as np

from .. import models
from ..simulation import run_simulations, predict_price_movement as _predict_price_movement


def predict_price_movement(token: str, *, model_path: str | None = None) -> float:
    """Wrapper around :func:`simulation.predict_price_movement`."""
    return _predict_price_movement(token, model_path=model_path)

from . import BaseAgent
from ..portfolio import Portfolio


class ConvictionAgent(BaseAgent):
    """Basic conviction calculation using expected ROI."""

    name = "conviction"

    def __init__(self, threshold: float = 0.05, count: int = 100, *, model_path: str | None = None):
        self.threshold = threshold
        self.count = count
        self.model_path = model_path or os.getenv("PRICE_MODEL_PATH")

    def _predict_return(self, token: str) -> float:
        return predict_price_movement(token, model_path=self.model_path)

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
        pred = self._predict_return(token)
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
