from __future__ import annotations

from typing import Any, Dict, List

from solhunter_zero.investor_demo import _buy_and_hold as buy_and_hold

# Simple strategy tuple consumed by the demo backtester
STRATEGY = ("buy_hold", buy_and_hold)


async def evaluate(token: str, portfolio: Any) -> List[Dict[str, Any]]:
    """Stub evaluate function so StrategyManager can load this module."""
    return []
