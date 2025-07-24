from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator, Iterable, List, Dict

from solders.keypair import Keypair

from . import BaseAgent
from .mev_sandwich import _fetch_swap_tx_message
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from ..mempool_scanner import stream_ranked_mempool_tokens_with_depth

stream_ranked_mempool_tokens_with_depth = None

from ..flash_loans import borrow_flash, repay_flash
from ..onchain_metrics import fetch_slippage_onchain
from ..depth_client import prepare_signed_tx
from ..mev_executor import MEVExecutor
from ..portfolio import Portfolio
from ..exchange import DEX_BASE_URL


class FlashloanSandwichAgent(BaseAgent):
    """Use flash loans to create sandwich bundles."""

    name = "flashloan_sandwich"

    def __init__(
        self,
        slippage_threshold: float = 0.2,
        *,
        size_threshold: float = 1.0,
        amount: float = 1.0,
        priority_rpc: Iterable[str] | None = None,
        jito_rpc_url: str | None = None,
        jito_auth: str | None = None,
        base_url: str = DEX_BASE_URL,
        payer: Keypair | None = None,
    ) -> None:
        self.slippage_threshold = float(slippage_threshold)
        self.size_threshold = float(size_threshold)
        self.amount = float(amount)
        self.priority_rpc = list(priority_rpc) if priority_rpc else None
        self.jito_rpc_url = jito_rpc_url or os.getenv("JITO_RPC_URL")
        self.jito_auth = jito_auth or os.getenv("JITO_AUTH")
        self.base_url = base_url
        self.payer = payer or Keypair()

    async def _prepare_tx(self, token: str, side: str, amount: float) -> str | None:
        msg = await _fetch_swap_tx_message(token, side, amount, 0.0, self.base_url)
        if not msg:
            return None
        return await prepare_signed_tx(msg)

    async def listen(
        self,
        rpc_url: str,
        *,
        suffix: str | None = None,
        keywords: Iterable[str] | None = None,
        include_pools: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Yield tokens that triggered a flash-loan sandwich bundle."""
        from ..mempool_scanner import (
            stream_ranked_mempool_tokens_with_depth as _default_stream,
        )

        stream_fn = stream_ranked_mempool_tokens_with_depth or _default_stream

        async for event in stream_fn(
            rpc_url,
            suffix=suffix,
            keywords=keywords,
            include_pools=include_pools,
        ):
            token = event["address"]
            size = float(event.get("avg_swap_size", 0.0))
            slip = await asyncio.to_thread(fetch_slippage_onchain, token, rpc_url)
            if slip >= self.slippage_threshold or size >= self.size_threshold:
                amt = max(self.amount, size)
                front = await self._prepare_tx(token, "buy", amt)
                back = await self._prepare_tx(token, "sell", amt)
                if front and back:
                    mev = MEVExecutor(
                        token,
                        priority_rpc=self.priority_rpc,
                        jito_rpc_url=self.jito_rpc_url,
                        jito_auth=self.jito_auth,
                    )
                    sig = await borrow_flash(amt, token, [], payer=self.payer)
                    await mev.submit_bundle([front, back])
                    if sig:
                        await repay_flash(sig)
                yield token

    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, any]]:
        # Trading decisions are handled in ``listen``
        return []
