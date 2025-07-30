# Utility functions for runtime helpers.

from __future__ import annotations

import asyncio
import importlib
import logging

logger = logging.getLogger(__name__)


def install_uvloop() -> None:
    """Install ``uvloop`` if available.

    Importing and calling this function sets ``asyncio``'s event loop
    policy to ``uvloop``. If the optional dependency is not installed
    the function silently does nothing.
    """
    try:
        uvloop = importlib.import_module("uvloop")
    except Exception:  # pragma: no cover - optional dependency missing
        logger.debug("uvloop not installed")
        return

    if getattr(asyncio, "_uvloop_installed", False):
        return

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio._uvloop_installed = True  # type: ignore[attr-defined]
    logger.debug("uvloop installed as event loop policy")
