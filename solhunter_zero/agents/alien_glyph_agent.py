from __future__ import annotations

from typing import Callable, Sequence, List, Dict, Any

from . import BaseAgent
from ..portfolio import Portfolio
from .. import alien_math


class AlienGlyphAgent(BaseAgent):
    """Analyze alien glyph patterns to decide trade actions."""

    name = "alien_glyph"

    def __init__(
        self,
        threshold: float = 0.0,
        glyph_loader: Callable[[str], Sequence[int]] | None = None,
    ) -> None:
        self.threshold = threshold
        self.glyph_loader = glyph_loader or (lambda token: [])

    def _score(self, glyphs: Sequence[int]) -> float:
        strength = alien_math.glyph_strength(glyphs)
        bias = alien_math.glyph_bias(glyphs)
        return strength + bias

    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        glyphs = self.glyph_loader(token)
        score = self._score(glyphs)
        if score > self.threshold:
            return [{"token": token, "side": "buy", "amount": 1.0, "price": 0.0}]
        if score < -self.threshold:
            pos = portfolio.balances.get(token)
            if pos:
                return [
                    {"token": token, "side": "sell", "amount": pos.amount, "price": 0.0}
                ]
        return []
