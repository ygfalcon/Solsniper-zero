from __future__ import annotations

from typing import Any, Dict, List

from . import BaseAgent
from ..portfolio import Portfolio


class EmotionAgent(BaseAgent):
    """Assign emotion tags to executed trades."""

    name = "emotion"

    def score(self, conviction_delta: float, regret: float, misfires: float) -> float:
        """Combine factors into a single score."""
        return conviction_delta - regret - misfires

    def evaluate(self, action: Dict[str, Any], result: Any) -> str:
        """Return an emotion label for a completed trade."""
        delta = float(action.get("conviction_delta", 0.0))
        regret = float(action.get("regret", 0.0))
        misfires = float(action.get("misfires", 0.0))
        score = self.score(delta, regret, misfires)
        if score > 0.5:
            return "confident"
        if score < -0.5:
            return "anxious"
        return "neutral"

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        # The emotion agent itself does not propose trades
        return []
