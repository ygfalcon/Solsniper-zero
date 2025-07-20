from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..portfolio import Portfolio


class BaseAgent(ABC):
    """Abstract trading agent."""

    name: str = "base"

    @abstractmethod
    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        """Return proposed trade actions for ``token``."""
        raise NotImplementedError
