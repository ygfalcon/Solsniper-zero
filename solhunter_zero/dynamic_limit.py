import os
import psutil

# Module level parameters read once on import.
_KP: float = float(os.getenv("CONCURRENCY_KP", "0.5") or 0.5)
_SMOOTHING: float = float(os.getenv("CONCURRENCY_SMOOTHING", "0.2") or 0.2)
_CPU_EMA: float = 0.0


def refresh_params() -> None:
    """Reload concurrency parameters from environment variables."""
    global _KP, _SMOOTHING
    _KP = float(os.getenv("CONCURRENCY_KP", str(_KP)) or _KP)
    _SMOOTHING = float(os.getenv("CONCURRENCY_SMOOTHING", str(_SMOOTHING)) or _SMOOTHING)


def _target_concurrency(cpu: float, base: int, low: float, high: float) -> int:
    """Return desired concurrency for ``cpu`` usage.

    The function smooths CPU usage using an exponential moving average and
    adjusts the target based on current memory pressure.
    """

    global _CPU_EMA

    smoothing = _SMOOTHING
    if _CPU_EMA:
        _CPU_EMA = smoothing * cpu + (1.0 - smoothing) * _CPU_EMA
    else:
        _CPU_EMA = cpu

    try:
        mem = float(psutil.virtual_memory().percent)
    except Exception:
        mem = 0.0

    load = max(_CPU_EMA, mem)

    if load <= low:
        return base
    if load >= high:
        return 1
    frac = (load - low) / (high - low)
    return max(1, int(round(base * (1.0 - frac))))


def _step_limit(current: int, target: int, max_val: int) -> int:
    """Move ``current`` towards ``target`` using :data:`_KP`."""

    kp = _KP
    new_val = current + kp * (target - current)
    new_val = int(round(new_val))
    if new_val > max_val:
        return max_val
    if new_val < 1:
        return 1
    return new_val
