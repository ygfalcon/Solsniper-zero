from __future__ import annotations

from typing import List, Dict, Any

import os
import numpy as np

from .. import models
from ..simulation import run_simulations, fetch_token_metrics

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
        if not self.model_path:
            return 0.0
        model = models.get_model(self.model_path, reload=True)
        if not model:
            return 0.0
        metrics = fetch_token_metrics(token)
        ph = metrics.get("price_history") or []
        lh = metrics.get("liquidity_history") or []
        dh = metrics.get("depth_history") or []
        sh = metrics.get("slippage_history") or []
        vh = metrics.get("volume_history") or []
        th = metrics.get("tx_count_history") or []
        n = min(len(ph), len(lh), len(dh), len(sh or ph), len(vh or ph), len(th or ph))
        if n < 30:
            return 0.0
        seq = np.column_stack([
            ph[-30:],
            lh[-30:],
            dh[-30:],
            (sh or [0] * n)[-30:],
            (vh or [0] * n)[-30:],
            (th or [0] * n)[-30:],
        ])
        try:
            return float(model.predict(seq))
        except Exception:
            return 0.0

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
