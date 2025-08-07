from __future__ import annotations

from typing import Any, Dict, List

from solhunter_zero.investor_demo import _momentum as momentum

STRATEGY = ("momentum", momentum)


async def evaluate(token: str, portfolio: Any) -> List[Dict[str, Any]]:
    """Stub evaluate function so StrategyManager can load this module."""
    return []
