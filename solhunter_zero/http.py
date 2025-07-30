from __future__ import annotations

import aiohttp
from .rpc import close_rpc_clients

_session: aiohttp.ClientSession | None = None

async def get_session() -> aiohttp.ClientSession:
    """Return a shared :class:`aiohttp.ClientSession`."""
    global _session
    if _session is None or getattr(_session, "closed", False):
        _session = aiohttp.ClientSession()
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
    try:
        await close_rpc_clients()
    except Exception:
        pass
