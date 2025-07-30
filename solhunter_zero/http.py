from __future__ import annotations

import os
import aiohttp

_DEFAULT_CONN_LIMIT = int(os.getenv("HTTP_CONN_LIMIT", "100") or 100)
_DEFAULT_KEEPALIVE = float(os.getenv("HTTP_KEEPALIVE", "60") or 60)

_session: aiohttp.ClientSession | None = None

async def get_session(
    conn_limit: int | None = None,
    keepalive_timeout: float | None = None,
) -> aiohttp.ClientSession:
    """Return a shared :class:`aiohttp.ClientSession`.

    Parameters can be overridden for the first call or via the environment
    variables ``HTTP_CONN_LIMIT`` and ``HTTP_KEEPALIVE``.
    """
    global _session
    if _session is None or getattr(_session, "closed", False):
        limit = conn_limit if conn_limit is not None else _DEFAULT_CONN_LIMIT
        keepalive = (
            keepalive_timeout if keepalive_timeout is not None else _DEFAULT_KEEPALIVE
        )
        connector = aiohttp.TCPConnector(
            limit=limit,
            keepalive_timeout=keepalive,
        )
        _session = aiohttp.ClientSession(connector=connector)
    return _session

async def close_session() -> None:
    """Close the shared session if it exists."""
    global _session
    if _session is not None:
        close = getattr(_session, "close", None)
        if callable(close) and not getattr(_session, "closed", False):
            await close()
    _session = None
    try:
        from .depth_client import close_mmap, close_ipc_clients
        close_mmap()
        await close_ipc_clients()
    except Exception:
        pass
