import os

_KP: float = float(os.getenv("CONCURRENCY_KP", "0.5") or 0.5)


def _target_concurrency(cpu: float, base: int, low: float, high: float) -> int:
    """Return desired concurrency for ``cpu`` usage."""
    if cpu <= low:
        return base
    if cpu >= high:
        return 1
    frac = (cpu - low) / (high - low)
    return max(1, int(round(base * (1.0 - frac))))


def _step_limit(current: int, target: int, max_val: int) -> int:
    """Move ``current`` towards ``target`` using :data:`_KP`."""
    new_val = current + _KP * (target - current)
    new_val = int(round(new_val))
    if new_val > max_val:
        return max_val
    if new_val < 1:
        return 1
    return new_val
