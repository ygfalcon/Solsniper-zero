from __future__ import annotations

"""Aggregate system metrics from multiple sources."""

import asyncio
from collections import deque
from typing import Any, Deque, Tuple

from .event_bus import publish, subscription

# Keep the last few readings
_HISTORY: Deque[Tuple[float, float]] = deque(maxlen=4)
# running totals for O(1) average updates
_CPU_TOTAL: float = 0.0
_MEM_TOTAL: float = 0.0


def _on_metrics(msg: Any) -> None:
    """Handle incoming ``system_metrics`` events and publish an average."""
    cpu = getattr(msg, "cpu", None)
    mem = getattr(msg, "memory", None)
    if cpu is None and isinstance(msg, dict):
        cpu = msg.get("cpu")
        mem = msg.get("memory")
    if cpu is None or mem is None:
        return
    try:
        cpu = float(cpu)
        mem = float(mem)
    except Exception:
        return
    global _CPU_TOTAL, _MEM_TOTAL
    if len(_HISTORY) == _HISTORY.maxlen:
        old_cpu, old_mem = _HISTORY.popleft()
        _CPU_TOTAL -= old_cpu
        _MEM_TOTAL -= old_mem
    _HISTORY.append((cpu, mem))
    _CPU_TOTAL += cpu
    _MEM_TOTAL += mem
    avg_cpu = _CPU_TOTAL / len(_HISTORY)
    avg_mem = _MEM_TOTAL / len(_HISTORY)
    publish("system_metrics_combined", {"cpu": avg_cpu, "memory": avg_mem})


_def_sub = None


def start() -> None:
    """Begin aggregating ``system_metrics`` events."""
    global _def_sub
    if _def_sub is None:
        _def_sub = subscription("system_metrics", _on_metrics)
        _def_sub.__enter__()


async def _run_forever() -> None:
    start()
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":  # pragma: no cover - manual execution
    asyncio.run(_run_forever())
