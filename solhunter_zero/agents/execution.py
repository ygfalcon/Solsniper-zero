from __future__ import annotations

import asyncio
import base64
from typing import Dict, Any, List

import aiohttp

from . import BaseAgent
from ..exchange import (
    place_order_async,
    ORCA_DEX_URL,
    RAYDIUM_DEX_URL,
    DEX_BASE_URL,
    VENUE_URLS,
    SWAP_PATH,
    _sign_transaction,
)
from ..execution import EventExecutor
from ..depth_client import submit_raw_tx
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
        retries: int = 1,
        depth_service: bool = False,
        priority_rpc: list[str] | None = None,
    ):
        self.rate_limit = rate_limit
        self.testnet = testnet
        self.dry_run = dry_run
        self.keypair = keypair
        self.retries = retries
        self._sem = asyncio.Semaphore(concurrency)
        self._rate_lock = asyncio.Lock()
        self._last = 0.0
        self.depth_service = depth_service
        self._executors: Dict[str, EventExecutor] = {}
        self.priority_rpc = list(priority_rpc) if priority_rpc else None

    async def _create_signed_tx(
        self,
        token: str,
        side: str,
        amount: float,
        price: float,
        base_url: str,
    ) -> str | None:
        """Return a signed transaction for ``token`` using ``base_url``."""

        payload = {
            "token": token,
            "side": side,
            "amount": amount,
            "price": price,
            "cluster": "devnet" if self.testnet else "mainnet-beta",
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{base_url}{SWAP_PATH}", json=payload, timeout=10
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            except aiohttp.ClientError:
                return None

        tx_b64 = data.get("swapTransaction")
        if not tx_b64 or self.keypair is None:
            return None

        tx = _sign_transaction(tx_b64, self.keypair)
        return base64.b64encode(bytes(tx)).decode()

    def add_executor(self, token: str, executor: EventExecutor) -> None:
        """Register an :class:`EventExecutor` for ``token``."""

        self._executors[token] = executor

    async def execute(self, action: Dict[str, Any]) -> Any:
        async with self._sem:
            async with self._rate_lock:
                now = asyncio.get_event_loop().time()
                delay = self.rate_limit - (now - self._last)
                if delay > 0:
                    await asyncio.sleep(delay)
                self._last = asyncio.get_event_loop().time()

            venue = str(action.get("venue", "")).lower()
            venues = action.get("venues")

            if venues and isinstance(venues, list):
                base_urls = [VENUE_URLS.get(v, v) for v in venues]
            else:
                if venue == "orca":
                    base_urls = [ORCA_DEX_URL]
                elif venue == "raydium":
                    base_urls = [RAYDIUM_DEX_URL]
                else:
                    base_urls = [DEX_BASE_URL]

            if self.depth_service:
                for url in base_urls:
                    tx = await self._create_signed_tx(
                        action["token"],
                        action["side"],
                        action.get("amount", 0.0),
                        action.get("price", 0.0),
                        url,
                    )
                    if tx:
                        execer = self._executors.get(action["token"])
                        if execer:
                            await execer.enqueue(tx)
                        else:
                            await submit_raw_tx(
                                tx,
                                priority_rpc=self.priority_rpc,
                            )
                        return {"queued": True}
                return None

            return await place_order_async(
                action["token"],
                action["side"],
                action.get("amount", 0.0),
                action.get("price", 0.0),
                testnet=self.testnet,
                dry_run=self.dry_run,
                keypair=self.keypair,
                base_url=base_urls[0],
                venues=venues,
                max_retries=action.get("retries", self.retries),
                timeout=action.get("timeout"),
            )

    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        # Execution agent does not propose trades itself
        return []
