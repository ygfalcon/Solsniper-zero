from __future__ import annotations

import os
import aiohttp

try:
    import orjson as _json  # type: ignore
    USE_ORJSON = True
except Exception:  # pragma: no cover - optional dependency
    import json as _json  # type: ignore
    USE_ORJSON = False


def dumps(obj: object) -> str | bytes:
    """Serialize *obj* to JSON using ``orjson`` when available."""
    if USE_ORJSON:
        return _json.dumps(obj)
    return _json.dumps(obj)


def loads(data: str | bytes) -> object:
    """Deserialize JSON *data* using ``orjson`` when available."""
    if USE_ORJSON and isinstance(data, str):
        data = data.encode()
    return _json.loads(data)

_session: aiohttp.ClientSession | None = None

# Connector limits are configurable via environment variables.
CONNECTOR_LIMIT = int(os.getenv("HTTP_CONNECTOR_LIMIT", "0") or 0)
CONNECTOR_LIMIT_PER_HOST = int(os.getenv("HTTP_CONNECTOR_LIMIT_PER_HOST", "0") or 0)

async def get_session() -> aiohttp.ClientSession:
    """Return a shared :class:`aiohttp.ClientSession`."""
    global _session
    if _session is None or getattr(_session, "closed", False):
        connector = aiohttp.TCPConnector(
            limit=CONNECTOR_LIMIT, limit_per_host=CONNECTOR_LIMIT_PER_HOST
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
