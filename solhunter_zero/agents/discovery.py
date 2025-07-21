from __future__ import annotations

import os
from typing import List

from . import BaseAgent
from ..async_scanner import scan_tokens_async
from ..portfolio import Portfolio


class DiscoveryAgent(BaseAgent):
    """Discover tokens using existing scanners."""

    name = "discovery"

    async def discover_tokens(self, *, offline: bool = False, token_file: str | None = None, method: str | None = None) -> List[str]:
        if method is None:
            method = os.getenv("DISCOVERY_METHOD", "websocket")
        return await scan_tokens_async(offline=offline, token_file=token_file, method=method)

    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ):
        # Discovery agent does not propose trades for individual tokens
        return []
