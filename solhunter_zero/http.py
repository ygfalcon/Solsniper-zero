from __future__ import annotations

import aiohttp

_session: aiohttp.ClientSession | None = None

async def get_session() -> aiohttp.ClientSession:
    """Return a shared :class:`aiohttp.ClientSession`."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session

async def close_session() -> None:
    """Close the shared session if it exists."""
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None
