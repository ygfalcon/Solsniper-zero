from __future__ import annotations

import os
import aiohttp

try:
    import orjson as _json  # type: ignore
    USE_ORJSON = True
except Exception:  # pragma: no cover - optional dependency
    import json as _json  # type: ignore
    USE_ORJSON = False


def dumps(obj: object) -> bytes:
    """Serialize *obj* to JSON bytes using ``orjson`` when available."""
    if USE_ORJSON:
        return _json.dumps(obj)
    return _json.dumps(obj).encode()


def loads(data: str | bytes) -> object:
    """Deserialize JSON *data* using ``orjson`` when available."""
    if USE_ORJSON:
        if isinstance(data, str):
            data = data.encode()
        return _json.loads(data)
    if isinstance(data, bytes):
        data = data.decode()
    return _json.loads(data)


def check_endpoint(url: str, retries: int = 3) -> None:
    """Send a ``HEAD`` request to *url* ensuring it is reachable.

    The request is attempted up to ``retries`` times using exponential backoff
    (1s, 2s, ...).  If all attempts fail a :class:`urllib.error.URLError` is
    raised.
    """

    import time
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, method="HEAD")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=5):  # nosec B310
                return
        except urllib.error.URLError as exc:  # pragma: no cover - network failure
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            print(
                f"Attempt {attempt + 1} failed for {url}: {exc}. Retrying in {wait} seconds..."
            )
            time.sleep(wait)

_session: aiohttp.ClientSession | None = None

# Connector limits are configurable via environment variables.
CONNECTOR_LIMIT = int(os.getenv("HTTP_CONNECTOR_LIMIT", "0") or 0)
CONNECTOR_LIMIT_PER_HOST = int(os.getenv("HTTP_CONNECTOR_LIMIT_PER_HOST", "0") or 0)

async def get_session() -> aiohttp.ClientSession:
    """Return a shared :class:`aiohttp.ClientSession`."""
    global _session
    if _session is None or getattr(_session, "closed", False):
        conn_cls = getattr(aiohttp, "TCPConnector", None)
        if conn_cls is object or conn_cls is None:
            connector = None
        else:
            connector = conn_cls(
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
