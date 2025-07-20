from __future__ import annotations

from typing import List, Dict, Any

from . import BaseAgent
from ..memory import Memory
from ..portfolio import Portfolio


class MemoryAgent(BaseAgent):
    """Record executed trades and classify outcomes."""

    name = "memory"

    def __init__(self, memory: Memory | None = None):
        self.memory = memory or Memory("sqlite:///:memory:")

    async def log(self, action: Dict[str, Any]) -> None:
        self.memory.log_trade(
            token=action.get("token"),
            direction=action.get("side"),
            amount=action.get("amount", 0.0),
            price=action.get("price", 0.0),
            reason=action.get("agent"),
        )

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        # Memory agent does not produce trades
        return []
