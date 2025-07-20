from __future__ import annotations

import asyncio
from typing import Dict, Any, List

from . import BaseAgent
from ..exchange import place_order_async
from ..portfolio import Portfolio


class ExecutionAgent(BaseAgent):
    """Submit orders with simple rate limiting."""

    name = "execution"

    def __init__(self, *, rate_limit: float = 1.0, testnet: bool = False, dry_run: bool = False, keypair=None):
        self.rate_limit = rate_limit
        self.testnet = testnet
        self.dry_run = dry_run
        self.keypair = keypair
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def execute(self, action: Dict[str, Any]) -> Any:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            delay = self.rate_limit - (now - self._last)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = asyncio.get_event_loop().time()
        return await place_order_async(
            action["token"],
            action["side"],
            action.get("amount", 0.0),
            action.get("price", 0.0),
            testnet=self.testnet,
            dry_run=self.dry_run,
            keypair=self.keypair,
        )

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        # Execution agent does not propose trades itself
        return []
