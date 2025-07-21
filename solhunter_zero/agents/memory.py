from __future__ import annotations

from typing import List, Dict, Any

from . import BaseAgent
from ..memory import Memory
from ..advanced_memory import AdvancedMemory
from ..portfolio import Portfolio


class MemoryAgent(BaseAgent):
    """Record executed trades and classify outcomes."""

    name = "memory"

    def __init__(self, memory: Memory | AdvancedMemory | None = None):
        self.memory = memory or Memory("sqlite:///:memory:")

    async def log(self, action: Dict[str, Any], *, skip_db: bool = False) -> None:
        """Record ``action`` in memory unless ``skip_db`` is True."""
        if skip_db:
            return
        extra = {}
        if isinstance(self.memory, AdvancedMemory):
            extra = {
                "context": action.get("context", ""),
                "emotion": action.get("emotion", ""),
                "simulation_id": action.get("simulation_id"),
            }
        self.memory.log_trade(
            token=action.get("token"),
            direction=action.get("side"),
            amount=action.get("amount", 0.0),
            price=action.get("price", 0.0),
            reason=action.get("agent"),
            **extra,
        )

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        # Memory agent does not produce trades
        return []
