from __future__ import annotations

from typing import Any, Dict, List

from solhunter_zero.investor_demo import _mean_reversion


def mean_reversion(prices, liquidity=None):
    return _mean_reversion(prices)


STRATEGY = ("mean_reversion", mean_reversion)


async def evaluate(token: str, portfolio: Any) -> List[Dict[str, Any]]:
    """Stub evaluate function so StrategyManager can load this module."""
    return []
