"""Event-driven transaction execution loop."""

from __future__ import annotations

import asyncio

from .depth_client import stream_depth, submit_raw_tx, DEPTH_SERVICE_SOCKET


class EventExecutor:
    """Trigger transaction submission on depth/mempool updates."""

    def __init__(
        self,
        token: str,
        *,
        rate_limit: float = 0.05,
        threshold: float = 0.0,
        socket_path: str = DEPTH_SERVICE_SOCKET,
        priority_rpc: list[str] | None = None,
    ) -> None:
        self.token = token
        self.rate_limit = rate_limit
        self.threshold = threshold
        self.socket_path = socket_path
        self.priority_rpc = list(priority_rpc) if priority_rpc else None
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def enqueue(self, tx_b64: str) -> None:
        """Queue a pre-signed transaction for immediate submission."""

        await self._queue.put(tx_b64)

    async def run(self) -> None:
        """Start the event loop."""

        async for update in stream_depth(
            self.token, rate_limit=self.rate_limit
        ):
            rate = float(update.get("tx_rate", 0.0))
            if rate < self.threshold:
                continue
            try:
                tx = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                continue
            await submit_raw_tx(
                tx,
                socket_path=self.socket_path,
                priority_rpc=self.priority_rpc,
            )
            self._queue.task_done()


async def run_event_loop(
    token: str,
    tx_source: "asyncio.Queue[str] | asyncio.Queue[bytes]",
    *,
    rate_limit: float = 0.05,
    threshold: float = 0.0,
    socket_path: str = DEPTH_SERVICE_SOCKET,
    priority_rpc: list[str] | None = None,
) -> None:
    """Convenience wrapper around :class:`EventExecutor`."""

    execer = EventExecutor(
        token,
        rate_limit=rate_limit,
        threshold=threshold,
        socket_path=socket_path,
        priority_rpc=priority_rpc,
    )

    async def _feed() -> None:
        while True:
            tx = await tx_source.get()
            await execer.enqueue(tx if isinstance(tx, str) else tx.decode())
            tx_source.task_done()

    await asyncio.gather(execer.run(), _feed())

