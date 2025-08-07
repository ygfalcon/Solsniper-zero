#!/usr/bin/env python
"""Minimal one-click trading demo.

The script attempts to import optional strategy agents. Missing optional
dependencies should not prevent the demo from running; instead a warning is
logged for each unavailable feature.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional strategy agents
try:  # optional dependency
    from solhunter_zero.agents import smart_discovery as _sd
    SmartDiscoveryAgent = _sd.SmartDiscoveryAgent
    if getattr(_sd, "GradientBoostingRegressor", None) is None:
        raise ImportError("sklearn missing")
except Exception as exc:  # pragma: no cover - optional dependency
    SmartDiscoveryAgent = None  # type: ignore[assignment]
    log.warning("SmartDiscoveryAgent unavailable: %s", exc)

try:  # optional dependency
    from solhunter_zero.agents import strange_attractor as _sa
    StrangeAttractorAgent = _sa.StrangeAttractorAgent
    if getattr(_sa, "solve_ivp", None) is None or getattr(_sa, "faiss", None) is None:
        raise ImportError("scipy/faiss missing")
except Exception as exc:  # pragma: no cover - optional dependency
    StrangeAttractorAgent = None  # type: ignore[assignment]
    log.warning("StrangeAttractorAgent unavailable: %s", exc)


def main() -> int:
    """Entry point for the demo."""
    logging.basicConfig(level=logging.INFO)
    log.info("One-click trading demo ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
