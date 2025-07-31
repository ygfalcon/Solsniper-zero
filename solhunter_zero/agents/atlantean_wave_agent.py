from __future__ import annotations

import math
from typing import List, Dict, Any

from . import BaseAgent
from ..portfolio import Portfolio

from alien_math import pyramid_frequency


class AtlanteanWaveAgent(BaseAgent):
    """Model cyclical wave patterns inspired by lost civilizations."""

    name = "atlantean_wave"

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = float(threshold)

    def _wave_series(self, depth: float, imbalance: float) -> List[float]:
        series = []
        for i in range(10):
            val = depth * math.sin(i) + imbalance * math.cos(i / 2.0)
            series.append(val)
        return series

    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        series = self._wave_series(float(depth or 0.0), float(imbalance or 0.0))
        try:
            freq = float(pyramid_frequency(series))
        except Exception:
            freq = 0.0

        if freq >= self.threshold:
            return [
                {
                    "token": token,
                    "side": "buy",
                    "amount": 1.0,
                    "price": 0.0,
                    "frequency": freq,
                }
            ]
        return []
