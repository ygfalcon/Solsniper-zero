from __future__ import annotations

import asyncio
from typing import Dict

from solana.rpc.api import Client  # type: ignore
from solana.rpc.async_api import AsyncClient  # type: ignore

_SYNC_CLIENTS: Dict[str, Client] = {}
_ASYNC_CLIENTS: Dict[str, AsyncClient] = {}


def get_rpc_client(rpc_url: str):
    """Return a cached ``Client`` or ``AsyncClient`` for ``rpc_url``."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        client = _SYNC_CLIENTS.get(rpc_url)
        if client is None:
            client = Client(rpc_url)
            _SYNC_CLIENTS[rpc_url] = client
        return client
    else:
        client = _ASYNC_CLIENTS.get(rpc_url)
        if client is None or getattr(client, "closed", False) or getattr(client, "is_closed", False):
            client = AsyncClient(rpc_url)
            _ASYNC_CLIENTS[rpc_url] = client
        return client


async def close_rpc_clients() -> None:
    """Close all cached RPC clients and clear the caches."""
    global _SYNC_CLIENTS, _ASYNC_CLIENTS
    for client in list(_ASYNC_CLIENTS.values()):
        try:
            await client.close()
        except Exception:
            pass
    for client in list(_SYNC_CLIENTS.values()):
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    _ASYNC_CLIENTS.clear()
    _SYNC_CLIENTS.clear()
