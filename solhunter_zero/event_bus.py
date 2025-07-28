import asyncio
import inspect
import json
import os
from contextlib import contextmanager
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, Generator, List, Set

from .schemas import validate_message, to_dict


def _get_bus_url(cfg=None):
    from .config import get_event_bus_url
    return get_event_bus_url(cfg)

try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    websockets = None

# mapping of topic -> list of handlers
_subscribers: Dict[str, List[Callable[[Any], Awaitable[None] | None]]] = defaultdict(list)

# websocket related globals
_ws_clients: Set[Any] = set()
_ws_client = None
_ws_server = None

def subscribe(topic: str, handler: Callable[[Any], Awaitable[None] | None]):
    """Register ``handler`` for ``topic`` events.

    Returns a callable that will remove the handler when invoked.
    """
    _subscribers[topic].append(handler)

    def _unsub() -> None:
        unsubscribe(topic, handler)

    return _unsub


@contextmanager
def subscription(topic: str, handler: Callable[[Any], Awaitable[None] | None]) -> Generator[Callable[[Any], Awaitable[None] | None], None, None]:
    """Context manager that registers ``handler`` for ``topic`` and automatically unsubscribes."""
    unsub = subscribe(topic, handler)
    try:
        yield handler
    finally:
        unsub()

def unsubscribe(topic: str, handler: Callable[[Any], Awaitable[None] | None]):
    """Remove ``handler`` from ``topic`` subscriptions."""
    handlers = _subscribers.get(topic)
    if not handlers:
        return
    try:
        handlers.remove(handler)
    except ValueError:
        return
    if not handlers:
        _subscribers.pop(topic, None)

def publish(topic: str, payload: Any, *, _broadcast: bool = True) -> None:
    """Publish ``payload`` to all subscribers of ``topic`` and over websockets."""
    payload = validate_message(topic, payload)
    handlers = list(_subscribers.get(topic, []))
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    for h in handlers:
        if inspect.iscoroutinefunction(h):
            if loop:
                loop.create_task(h(payload))
            else:
                asyncio.run(h(payload))
        else:
            h(payload)

    if websockets and _broadcast:
        msg = json.dumps({"topic": topic, "payload": to_dict(payload)})
        if loop:
            loop.create_task(broadcast_ws(msg))
        else:
            asyncio.run(broadcast_ws(msg))


async def broadcast_ws(
    message: str, *, to_clients: bool = True, to_server: bool = True
) -> None:
    """Send ``message`` to websocket peers."""
    if to_clients:
        to_remove = []
        for ws in list(_ws_clients):
            try:
                await ws.send(message)
            except Exception:  # pragma: no cover - network errors
                to_remove.append(ws)
        for ws in to_remove:
            _ws_clients.discard(ws)
    if to_server and _ws_client is not None:
        try:
            await _ws_client.send(message)
        except Exception:  # pragma: no cover - connection issues
            pass


async def start_ws_server(host: str = "localhost", port: int = 8765):
    """Start websocket server broadcasting published events."""
    if not websockets:
        raise RuntimeError("websockets library required")

    async def handler(ws):
        _ws_clients.add(ws)
        try:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                    publish(
                        data.get("topic"), data.get("payload"), _broadcast=False
                    )
                except Exception:  # pragma: no cover - malformed message
                    continue
        finally:
            _ws_clients.discard(ws)

    global _ws_server
    _ws_server = await websockets.serve(handler, host, port)
    return _ws_server


async def stop_ws_server() -> None:
    """Stop the running websocket server and close client connections."""
    global _ws_server
    if _ws_server is not None:
        _ws_server.close()
        await _ws_server.wait_closed()
        _ws_server = None
    for ws in list(_ws_clients):
        try:
            await ws.close()
        except Exception:
            pass
    _ws_clients.clear()


async def connect_ws(url: str):
    """Connect to external websocket bus at ``url``."""
    if not websockets:
        raise RuntimeError("websockets library required")

    async def receiver(ws):
        async for msg in ws:
            try:
                data = json.loads(msg)
            except Exception:  # pragma: no cover - malformed msg
                continue
            publish(data.get("topic"), data.get("payload"), _broadcast=False)

    global _ws_client
    _ws_client = await websockets.connect(url)
    asyncio.create_task(receiver(_ws_client))
    return _ws_client


async def disconnect_ws() -> None:
    """Close websocket connection opened via ``connect_ws``."""
    global _ws_client
    if _ws_client is not None:
        try:
            await _ws_client.close()
        except Exception:
            pass
        _ws_client = None


# Automatically connect to an external event bus if configured
_ENV_URL = None


def _reload_bus(cfg) -> None:
    global _ENV_URL
    url = _get_bus_url(cfg)
    if url == _ENV_URL:
        return

    async def _reconnect() -> None:
        if _ws_client is not None:
            await disconnect_ws()
        if url:
            await connect_ws(url)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop:
        loop.create_task(_reconnect())
    else:
        asyncio.run(_reconnect())
    _ENV_URL = url


subscription("config_updated", _reload_bus).__enter__()

