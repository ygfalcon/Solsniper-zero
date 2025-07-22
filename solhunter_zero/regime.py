from __future__ import annotations
from typing import Sequence


def detect_regime(prices: Sequence[float], *, threshold: float = 0.02) -> str:
    """Return market regime label based on price trend.

    Parameters
    ----------
    prices:
        Historical price sequence ordered oldest to newest.
    threshold:
        Minimum fractional change over the window considered a trend.
    """
    if len(prices) < 2:
        return "sideways"
    change = prices[-1] / prices[0] - 1
    if change > threshold:
        return "bull"
    if change < -threshold:
        return "bear"
    return "sideways"
