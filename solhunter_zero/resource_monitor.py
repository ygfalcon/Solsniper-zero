from __future__ import annotations

import asyncio
import time
from typing import Optional

import psutil

from .event_bus import publish

_CPU_PERCENT: float = 0.0
_CPU_LAST: float = 0.0

_TASK: Optional[asyncio.Task] = None


async def _monitor(interval: float) -> None:
    """Publish system metrics every ``interval`` seconds."""
    try:
        while True:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            payload = {"cpu": float(cpu), "memory": float(mem)}
            global _CPU_PERCENT, _CPU_LAST
            _CPU_PERCENT = float(cpu)
            _CPU_LAST = time.monotonic()
            publish("system_metrics", payload)
            publish("remote_system_metrics", payload)
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


def get_cpu_usage() -> float:
    """Return the most recent CPU usage percentage."""
    global _CPU_PERCENT, _CPU_LAST
    if time.monotonic() - _CPU_LAST > 2.0:
        try:
            _CPU_PERCENT = float(psutil.cpu_percent(interval=None))
            _CPU_LAST = time.monotonic()
        except Exception:
            pass
    return _CPU_PERCENT


try:  # pragma: no cover - initialization best effort
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = None
if loop:
    loop.call_soon(start_monitor)
