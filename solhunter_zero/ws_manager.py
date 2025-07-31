from __future__ import annotations

import asyncio
from typing import Dict

import aiohttp

from .http import get_session

class WSConnectionManager:
    """Manage persistent WebSocket connections for DEX streams."""

    def __init__(self) -> None:
        self._sockets: Dict[str, aiohttp.ClientWebSocketResponse] = {}
        self._lock = asyncio.Lock()

    async def get_ws(self, url: str) -> aiohttp.ClientWebSocketResponse:
        """Return a connected WebSocket for ``url`` reopening if needed."""
        async with self._lock:
            ws = self._sockets.get(url)
            if ws is None or ws.closed:
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                session = await get_session()
                ws = await session.ws_connect(url)
                self._sockets[url] = ws
            return ws

    async def reconnect(self, url: str) -> aiohttp.ClientWebSocketResponse:
        """Force reconnection for ``url``."""
        await self.close(url)
        return await self.get_ws(url)

    async def close(self, url: str) -> None:
        """Close the WebSocket for ``url`` if open."""
        async with self._lock:
            ws = self._sockets.pop(url, None)
            if ws is not None and not ws.closed:
                try:
                    await ws.close()
                except Exception:
                    pass

    async def close_all(self) -> None:
        """Close all managed WebSockets."""
        async with self._lock:
            sockets = list(self._sockets.values())
            self._sockets.clear()
        for ws in sockets:
            if not ws.closed:
                try:
                    await ws.close()
                except Exception:
                    pass

WS_MANAGER = WSConnectionManager()
