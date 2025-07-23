from __future__ import annotations

from typing import Sequence, List, Optional

from .depth_client import submit_raw_tx, snapshot, DEPTH_SERVICE_SOCKET
from .gas import adjust_priority_fee


class MEVExecutor:
    """Bundle swap transactions and submit them with priority fees."""

    def __init__(
        self,
        token: str,
        *,
        priority_rpc: List[str] | None = None,
        socket_path: str = DEPTH_SERVICE_SOCKET,
    ) -> None:
        self.token = token
        self.priority_rpc = list(priority_rpc) if priority_rpc else None
        self.socket_path = socket_path

    async def submit_bundle(self, txs: Sequence[str]) -> List[Optional[str]]:
        """Submit ``txs`` with a compute unit price based on mempool rate."""

        _, rate = snapshot(self.token)
        priority_fee = adjust_priority_fee(rate)
        sigs = []
        for tx in txs:
            sig = await submit_raw_tx(
                tx,
                priority_rpc=self.priority_rpc,
                priority_fee=priority_fee,
                socket_path=self.socket_path,
            )
            sigs.append(sig)
        return sigs
