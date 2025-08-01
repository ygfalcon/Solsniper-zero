import os
import psutil

# Module level parameters read once on import.
_KP: float = float(os.getenv("CONCURRENCY_KP", "0.5") or 0.5)
_EWM_SMOOTHING: float = float(
    os.getenv("CONCURRENCY_EWM_SMOOTHING", "0.15") or 0.15
)
_CPU_EMA: float = 0.0


def refresh_params() -> None:
    """Reload concurrency parameters from environment variables."""
    global _KP, _EWM_SMOOTHING
    _KP = float(os.getenv("CONCURRENCY_KP", str(_KP)) or _KP)
    _EWM_SMOOTHING = float(
        os.getenv("CONCURRENCY_EWM_SMOOTHING", str(_EWM_SMOOTHING)) or _EWM_SMOOTHING
    )


def _target_concurrency(
    cpu: float,
    base: int,
    low: float,
    high: float,
    *,
    smoothing: float | None = None,
) -> int:
    """Return desired concurrency for ``cpu`` usage.

    The function smooths CPU usage using an exponential moving average and
    adjusts the target based on current memory pressure.
    """

    global _CPU_EMA

    if smoothing is None:
        smoothing = _EWM_SMOOTHING

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


def _step_limit(
    current: int, target: int, max_val: int, *, smoothing: float | None = None
) -> int:
    """Move ``current`` towards ``target`` using an exponential update."""

    if smoothing is None:
        smoothing = _KP
    new_val = current + smoothing * (target - current)
    new_val = int(round(new_val))
    if new_val > max_val:
        return max_val
    if new_val < 1:
        return 1
    return new_val
