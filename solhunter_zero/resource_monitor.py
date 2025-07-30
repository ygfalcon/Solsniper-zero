from __future__ import annotations

import asyncio
from typing import Optional

import psutil

from .event_bus import publish

_TASK: Optional[asyncio.Task] = None


async def _monitor(interval: float) -> None:
    """Publish CPU and memory usage every ``interval`` seconds."""
    try:
        while True:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            publish("resource_update", {"cpu": float(cpu), "memory": float(mem)})
            await asyncio.sleep(interval)
    except asyncio.CancelledError:  # pragma: no cover - cancellation
        pass


def start_monitor(interval: float = 1.0) -> asyncio.Task:
    """Start background resource monitoring task."""
    global _TASK
    if _TASK is None or _TASK.done():
        loop = asyncio.get_running_loop()
        _TASK = loop.create_task(_monitor(interval))
    return _TASK


def stop_monitor() -> None:
    """Stop the running resource monitor, if any."""
    global _TASK
    if _TASK is not None:
        _TASK.cancel()
        _TASK = None
