from __future__ import annotations

import asyncio
from typing import Dict, Any, List

from . import BaseAgent
from ..exchange import (
    place_order_async,
    ORCA_DEX_URL,
    RAYDIUM_DEX_URL,
    DEX_BASE_URL,
)
from ..portfolio import Portfolio


class ExecutionAgent(BaseAgent):
    """Submit orders with simple rate limiting."""

    name = "execution"

    def __init__(
        self,
        *,
        rate_limit: float = 1.0,
        concurrency: int = 1,
        testnet: bool = False,
        dry_run: bool = False,
        keypair=None,
    ):
        self.rate_limit = rate_limit
        self.testnet = testnet
        self.dry_run = dry_run
        self.keypair = keypair
        self._sem = asyncio.Semaphore(concurrency)
        self._rate_lock = asyncio.Lock()
        self._last = 0.0

    async def execute(self, action: Dict[str, Any]) -> Any:
        async with self._sem:
            async with self._rate_lock:
                now = asyncio.get_event_loop().time()
                delay = self.rate_limit - (now - self._last)
                if delay > 0:
                    await asyncio.sleep(delay)
                self._last = asyncio.get_event_loop().time()
            venue = str(action.get("venue", "")).lower()
            if venue == "orca":
                base_url = ORCA_DEX_URL
            elif venue == "raydium":
                base_url = RAYDIUM_DEX_URL
            else:
                base_url = DEX_BASE_URL
            return await place_order_async(
                action["token"],
                action["side"],
                action.get("amount", 0.0),
                action.get("price", 0.0),
                testnet=self.testnet,
                dry_run=self.dry_run,
                keypair=self.keypair,
                base_url=base_url,
            )

    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        # Execution agent does not propose trades itself
        return []
