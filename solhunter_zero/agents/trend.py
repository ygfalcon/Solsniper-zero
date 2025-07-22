from __future__ import annotations

from typing import List, Dict, Any, Iterable

from . import BaseAgent
from ..portfolio import Portfolio
from ..scanner_common import fetch_trending_tokens_async
from ..simulation import async_fetch_token_metrics
from ..news import fetch_sentiment


class TrendAgent(BaseAgent):
    """Buy trending tokens when volume and sentiment are strong."""

    name = "trend"

    def __init__(
        self,
        volume_threshold: float = 0.0,
        sentiment_threshold: float = 0.0,
        feeds: Iterable[str] | None = None,
    ) -> None:
        self.volume_threshold = volume_threshold
        self.sentiment_threshold = sentiment_threshold
        self.feeds = list(feeds) if feeds else []

    async def _current_sentiment(self) -> float:
        if not self.feeds:
            return 0.0
        try:
            return fetch_sentiment(self.feeds)
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
        trending = await fetch_trending_tokens_async()
        if token not in trending:
            return []
        metrics = await async_fetch_token_metrics(token)
        volume = float(metrics.get("volume", 0.0))
        sentiment = await self._current_sentiment()
        if volume >= self.volume_threshold and sentiment >= self.sentiment_threshold:
            return [
                {
                    "token": token,
                    "side": "buy",
                    "amount": 1.0,
                    "price": 0.0,
                    "volume": volume,
                    "sentiment": sentiment,
                }
            ]
        return []
