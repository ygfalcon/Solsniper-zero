from __future__ import annotations

from typing import List, Dict, Any

from . import BaseAgent
from ..portfolio import Portfolio
from ..ancient_math import decode_glyph, conviction_score


class ArtifactMathAgent(BaseAgent):
    """Conviction strategy using ancient glyph decoding heuristics."""

    name = "artifact_math"

    def __init__(self, threshold: float = 0.1, amount: float = 1.0) -> None:
        self.threshold = float(threshold)
        self.amount = float(amount)

    # ------------------------------------------------------------------
    def _score(self, token: str, depth: float | None, imbalance: float | None) -> float:
        glyphs = decode_glyph(token)
        return conviction_score(glyphs, depth=depth, imbalance=imbalance)

    # ------------------------------------------------------------------
    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        score = self._score(token, depth, imbalance)
        if score > self.threshold:
            return [
                {"token": token, "side": "buy", "amount": self.amount, "price": 0.0}
            ]
        if score < -self.threshold:
            pos = portfolio.balances.get(token)
            if pos:
                return [
                    {"token": token, "side": "sell", "amount": pos.amount, "price": 0.0}
                ]
        return []
