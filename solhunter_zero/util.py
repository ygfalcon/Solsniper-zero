# Utility functions for runtime helpers.

from __future__ import annotations

from typing import Any, Coroutine, TypeVar

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


T = TypeVar("T")


def run_coro(coro: Coroutine[Any, Any, T]) -> T | asyncio.Task:
    """Run ``coro`` using the active loop if present.

    When called with no running event loop this function falls back to
    :func:`asyncio.run` and returns the coroutine result.  If a loop is
    already running, the coroutine is scheduled with
    :func:`asyncio.AbstractEventLoop.create_task` and the resulting task
    is returned for awaiting by the caller.
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return loop.create_task(coro)
