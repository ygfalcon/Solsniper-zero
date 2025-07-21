from __future__ import annotations

from typing import Any, Dict, List, Iterable

from . import BaseAgent
from ..portfolio import Portfolio


from .. import news


class EmotionAgent(BaseAgent):
    """Assign emotion tags to executed trades and expose market sentiment."""

    name = "emotion"

    def __init__(self, feeds: Iterable[str] | None = None) -> None:
        self.feeds = list(feeds) if feeds else []
        self.sentiment: float = 0.0

    # ------------------------------------------------------------------
    def update_sentiment(self, allowed: Iterable[str] | None = None) -> float:
        """Refresh ``self.sentiment`` by querying configured news feeds."""
        if not self.feeds:
            self.sentiment = 0.0
            return self.sentiment
        try:
            self.sentiment = news.fetch_sentiment(self.feeds, allowed)
        except Exception:  # pragma: no cover - unexpected errors
            self.sentiment = 0.0
        return self.sentiment

    def score(
        self, conviction_delta: float, regret: float, misfires: float, sentiment: float = 0.0
    ) -> float:
        """Combine factors into a single score."""
        return conviction_delta - regret - misfires + sentiment

    def evaluate(self, action: Dict[str, Any], result: Any) -> str:
        """Return an emotion label for a completed trade."""
        delta = float(action.get("conviction_delta", 0.0))
        regret = float(action.get("regret", 0.0))
        misfires = float(action.get("misfires", 0.0))
        self.update_sentiment(self.feeds)
        score = self.score(delta, regret, misfires, self.sentiment)
        if score > 0.5:
            return "confident"
        if score < -0.5:
            return "anxious"
        return "neutral"

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        # The emotion agent itself does not propose trades
        return []
