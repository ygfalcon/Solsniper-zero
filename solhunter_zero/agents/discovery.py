from __future__ import annotations

import os
import asyncio
import logging
from typing import List, AsyncGenerator, Dict, Any, Iterable

from . import BaseAgent
from ..token_scanner import scan_tokens_async
from ..mempool_scanner import stream_ranked_mempool_tokens_with_depth
from ..scanner_common import SOLANA_RPC_URL
from ..discovery import merge_sources
from ..portfolio import Portfolio


logger = logging.getLogger(__name__)


class DiscoveryAgent(BaseAgent):
    """Discover tokens using existing scanners."""

    name = "discovery"

    def __init__(self) -> None:
        self.metrics: Dict[str, Dict[str, float]] = {}

    async def discover_tokens(
        self,
        *,
        offline: bool = False,
        token_file: str | None = None,
        method: str | None = None,
    ) -> List[str]:
        if method is None:
            method = os.getenv("DISCOVERY_METHOD", "websocket")

        async def _discover() -> List[str]:
            if not offline and token_file is None and method == "websocket":
                th = float(os.getenv("MEMPOOL_SCORE_THRESHOLD", "0") or 0.0)
                data = await merge_sources(SOLANA_RPC_URL, mempool_threshold=th)
                fields = [
                    "volume",
                    "liquidity",
                    "score",
                    "momentum",
                    "anomaly",
                    "wallet_concentration",
                    "avg_swap_size",
                ]
                self.metrics = {
                    d["address"]: {k: float(d.get(k, 0.0)) for k in fields} for d in data
                }
                return [d["address"] for d in data]

            tokens = await scan_tokens_async(
                offline=offline,
                token_file=token_file,
                method=method,
                dynamic_concurrency=True,
            )
            self.metrics = {t: {} for t in tokens}
            return tokens

        backoff = float(os.getenv("TOKEN_DISCOVERY_BACKOFF", "1") or 1.0)

        tokens = await _discover()
        if tokens:
            return tokens

        logger.warning("No tokens discovered; retrying in %s seconds", backoff)
        await asyncio.sleep(backoff)
        return await _discover()

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

    async def stream_mempool_events(
        self,
        rpc_url: str,
        *,
        threshold: float = 0.0,
        suffix: str | None = None,
        keywords: Iterable[str] | None = None,
        include_pools: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream ranked mempool events for immediate simulation."""

        async for event in stream_ranked_mempool_tokens_with_depth(
            rpc_url,
            suffix=suffix,
            keywords=keywords,
            include_pools=include_pools,
            threshold=threshold,
        ):
            yield event
