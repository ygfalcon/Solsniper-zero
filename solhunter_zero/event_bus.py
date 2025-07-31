import asyncio
import inspect

from .util import install_uvloop

install_uvloop()

try:
    import orjson as json  # type: ignore
    _USE_ORJSON = True
except Exception:  # pragma: no cover - optional dependency
    import json  # type: ignore
    _USE_ORJSON = False
import os
import zlib
from contextlib import contextmanager
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, Generator, List, Set


from .schemas import validate_message, to_dict
from . import event_pb2 as pb

_PB_MAP = {
    "action_executed": pb.ActionExecuted,
    "weights_updated": pb.WeightsUpdated,
    "rl_weights": pb.RLWeights,
    "rl_checkpoint": pb.RLCheckpoint,
    "portfolio_updated": pb.PortfolioUpdated,
    "depth_update": pb.DepthUpdate,
    "depth_service_status": pb.DepthServiceStatus,
    "heartbeat": pb.Heartbeat,
    "trade_logged": pb.TradeLogged,
    "rl_metrics": pb.RLMetrics,
    "system_metrics": pb.SystemMetrics,
    "price_update": pb.PriceUpdate,
}

# compress protobuf messages when broadcasting if enabled
_COMPRESS_EVENTS = os.getenv("COMPRESS_EVENTS")
COMPRESS_EVENTS = _COMPRESS_EVENTS not in (None, "", "0")


def _maybe_decompress(data: bytes) -> bytes:
    """Return decompressed ``data`` if it appears to be zlib-compressed."""
    if len(data) > 2 and data[0] == 0x78:
        try:
            return zlib.decompress(data)
        except Exception:
            return data
    return data


def _dumps(obj: Any) -> str | bytes:
    if _USE_ORJSON:
        return json.dumps(obj).decode()
    return json.dumps(obj)


def _loads(data: Any) -> Any:
    if _USE_ORJSON and isinstance(data, str):
        data = data.encode()
    return json.loads(data)


def _get_bus_url(cfg=None):
    from .config import get_event_bus_url
    return get_event_bus_url(cfg)

try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    websockets = None

# compression algorithm for websockets
_WS_COMPRESSION: str | None = os.getenv("EVENT_BUS_COMPRESSION", "deflate")
if _WS_COMPRESSION:
    comp = _WS_COMPRESSION.lower()
    if comp in {"", "none", "0"}:
        _WS_COMPRESSION = None

# mapping of topic -> list of handlers
_subscribers: Dict[str, List[Callable[[Any], Awaitable[None] | None]]] = defaultdict(list)

# websocket related globals
_ws_clients: Set[Any] = set()
_ws_client = None
_ws_server = None
_ws_url: str | None = None
_watch_task = None


def _encode_event(topic: str, payload: Any) -> Any:
    cls = _PB_MAP.get(topic)
    if cls is None:
        return _dumps({"topic": topic, "payload": to_dict(payload)})

    if topic == "action_executed":
        event = pb.Event(
            topic=topic,
            action_executed=pb.ActionExecuted(
                action_json=_dumps(payload.action),
                result_json=_dumps(payload.result),
            ),
        )
    elif topic == "weights_updated":
        event = pb.Event(topic=topic, weights_updated=pb.WeightsUpdated(weights=payload.weights))
    elif topic == "rl_weights":
        event = pb.Event(
            topic=topic,
            rl_weights=pb.RLWeights(weights=payload.weights, risk=payload.risk or {}),
        )
    elif topic == "rl_checkpoint":
        event = pb.Event(topic=topic, rl_checkpoint=pb.RLCheckpoint(time=payload.time, path=payload.path))
    elif topic == "portfolio_updated":
        event = pb.Event(topic=topic, portfolio_updated=pb.PortfolioUpdated(balances=payload.balances))
    elif topic == "depth_update":
        entries = {}
        for token, entry in to_dict(payload).items():
            if not isinstance(entry, dict):
                continue
            dex_map = entry.get("dex") or {
                k: v for k, v in entry.items() if isinstance(v, dict)
            }
            dex = {
                str(d): pb.TokenInfo(
                    bids=float(info.get("bids", 0.0)),
                    asks=float(info.get("asks", 0.0)),
                    tx_rate=float(info.get("tx_rate", 0.0)),
                )
                for d, info in dex_map.items()
                if isinstance(info, dict)
            }
            bids = float(entry.get("bids", 0.0))
            asks = float(entry.get("asks", 0.0))
            if not bids and not asks and dex:
                bids = sum(i.bids for i in dex.values())
                asks = sum(i.asks for i in dex.values())
            entries[str(token)] = pb.TokenAgg(
                dex=dex,
                bids=bids,
                asks=asks,
                tx_rate=float(entry.get("tx_rate", 0.0)),
                ts=int(entry.get("ts", 0)),
            )
        event = pb.Event(topic=topic, depth_update=pb.DepthUpdate(entries=entries))
        data = event.SerializeToString()
        if COMPRESS_EVENTS:
            data = zlib.compress(data)
        return data
    elif topic == "depth_service_status":
        event = pb.Event(topic=topic, depth_service_status=pb.DepthServiceStatus(status=payload.get("status")))
    elif topic == "heartbeat":
        service = getattr(payload, "service", None)
        if service is None and isinstance(payload, dict):
            service = payload.get("service")
        event = pb.Event(topic=topic, heartbeat=pb.Heartbeat(service=service or ""))
    elif topic == "trade_logged":
        data = to_dict(payload)
        event = pb.Event(
            topic=topic,
            trade_logged=pb.TradeLogged(
                token=str(data.get("token", "")),
                direction=str(data.get("direction", "")),
                amount=float(data.get("amount", 0.0)),
                price=float(data.get("price", 0.0)),
                reason=str(data.get("reason", "")),
                context=str(data.get("context", "")),
                emotion=str(data.get("emotion", "")),
                simulation_id=int(data.get("simulation_id", 0) or 0),
                uuid=str(data.get("uuid", "")),
                trade_id=int(data.get("trade_id", 0) or 0),
            ),
        )
    elif topic == "rl_metrics":
        data = to_dict(payload)
        event = pb.Event(
            topic=topic,
            rl_metrics=pb.RLMetrics(
                loss=float(data.get("loss", 0.0)),
                reward=float(data.get("reward", 0.0)),
            ),
        )
    elif topic == "system_metrics":
        data = to_dict(payload)
        event = pb.Event(
            topic=topic,
            system_metrics=pb.SystemMetrics(
                cpu=float(data.get("cpu", 0.0)),
                memory=float(data.get("memory", 0.0)),
            ),
        )
    elif topic == "price_update":
        data = to_dict(payload)
        event = pb.Event(
            topic=topic,
            price_update=pb.PriceUpdate(
                venue=str(data.get("venue", "")),
                token=str(data.get("token", "")),
                price=float(data.get("price", 0.0)),
            ),
        )
    else:
        return _dumps({"topic": topic, "payload": to_dict(payload)})
    data = event.SerializeToString()
    if COMPRESS_EVENTS:
        data = zlib.compress(data)
    return data

def _decode_payload(ev: pb.Event) -> Any:
    field = ev.WhichOneof("kind")
    if not field:
        return None
    msg = getattr(ev, field)
    if field == "action_executed":
        return {
            "action": _loads(msg.action_json),
            "result": _loads(msg.result_json),
        }
    if field == "weights_updated":
        return {"weights": dict(msg.weights)}
    if field == "rl_weights":
        return {"weights": dict(msg.weights), "risk": dict(msg.risk)}
    if field == "rl_checkpoint":
        return {"time": msg.time, "path": msg.path}
    if field == "portfolio_updated":
        return {"balances": dict(msg.balances)}
    if field == "depth_update":
        result = {}
        for token, entry in msg.entries.items():
            dex = {
                dk: {"bids": di.bids, "asks": di.asks, "tx_rate": di.tx_rate}
                for dk, di in entry.dex.items()
            }
            result[token] = {
                "dex": dex,
                "bids": entry.bids,
                "asks": entry.asks,
                "tx_rate": entry.tx_rate,
                "ts": entry.ts,
                "depth": entry.bids + entry.asks,
            }
        return result
    if field == "depth_service_status":
        return {"status": msg.status}
    if field == "heartbeat":
        return {"service": msg.service}
    if field == "trade_logged":
        return {
            "token": msg.token,
            "direction": msg.direction,
            "amount": msg.amount,
            "price": msg.price,
            "reason": msg.reason,
            "context": msg.context,
            "emotion": msg.emotion,
            "simulation_id": msg.simulation_id,
            "uuid": msg.uuid,
            "trade_id": msg.trade_id,
        }
    if field == "rl_metrics":
        return {"loss": msg.loss, "reward": msg.reward}
    if field == "system_metrics":
        return {"cpu": msg.cpu, "memory": msg.memory}
    if field == "price_update":
        return {
            "venue": msg.venue,
            "token": msg.token,
            "price": msg.price,
        }
    return None

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
        msg = _encode_event(topic, payload)
        if loop:
            loop.create_task(broadcast_ws(msg))
        else:
            asyncio.run(broadcast_ws(msg))


async def broadcast_ws(
    message: Any,
    *,
    to_clients: bool = True,
    to_server: bool = True,
) -> None:
    """Send ``message`` to websocket peers."""
    if to_clients:
        clients = list(_ws_clients)
        coros = [ws.send(message) for ws in clients]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for ws, res in zip(clients, results):
            if isinstance(res, Exception):  # pragma: no cover - network errors
                _ws_clients.discard(ws)
    if to_server and _ws_client is not None:
        try:
            await _ws_client.send(message)
        except Exception:  # pragma: no cover - connection issues
            pass


async def _receiver(ws) -> None:
    """Receive messages from ``ws`` and publish them."""
    try:
        async for msg in ws:
            try:
                if isinstance(msg, bytes):
                    data = _maybe_decompress(msg)
                    ev = pb.Event()
                    ev.ParseFromString(data)
                    payload = _decode_payload(ev)
                    publish(ev.topic, payload, _broadcast=False)
                    await broadcast_ws(msg, to_server=False)
                else:
                    data = _loads(msg)
                    topic = data.get("topic")
                    payload = data.get("payload")
                    publish(topic, payload, _broadcast=False)
                    await broadcast_ws(msg, to_server=False)
            except Exception:  # pragma: no cover - malformed msg
                continue
    finally:
        if ws is _ws_client and _ws_url:
            asyncio.create_task(reconnect_ws())


async def start_ws_server(host: str = "localhost", port: int = 8765):
    """Start websocket server broadcasting published events."""
    if not websockets:
        raise RuntimeError("websockets library required")

    async def handler(ws):
        _ws_clients.add(ws)
        try:
            async for msg in ws:
                try:
                    if isinstance(msg, bytes):
                        data = _maybe_decompress(msg)
                        ev = pb.Event()
                        ev.ParseFromString(data)
                        publish(ev.topic, _decode_payload(ev), _broadcast=False)
                    else:
                        data = _loads(msg)
                        publish(data.get("topic"), data.get("payload"), _broadcast=False)
                except Exception:  # pragma: no cover - malformed message
                    continue
        finally:
            _ws_clients.discard(ws)

    global _ws_server
    _ws_server = await websockets.serve(
        handler, host, port, compression=_WS_COMPRESSION
    )
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
    global _ws_client, _ws_url, _watch_task

    _ws_url = url
    _ws_client = await websockets.connect(url, compression=_WS_COMPRESSION)
    asyncio.create_task(_receiver(_ws_client))
    if _watch_task is None or _watch_task.done():
        loop = asyncio.get_running_loop()
        _watch_task = loop.create_task(_watch_ws())
    return _ws_client


async def disconnect_ws() -> None:
    """Close websocket connection opened via ``connect_ws``."""
    global _ws_client, _ws_url, _watch_task
    if _ws_client is not None:
        try:
            await _ws_client.close()
        except Exception:
            pass
        _ws_client = None
    _ws_url = None
    if _watch_task is not None:
        _watch_task.cancel()
        _watch_task = None


async def reconnect_ws(url: str | None = None) -> None:
    """(Re)connect the websocket client with exponential backoff."""
    global _ws_client, _ws_url, _watch_task
    if url:
        _ws_url = url
    if not _ws_url or not websockets:
        return
    backoff = 1.0
    max_backoff = 30
    while True:
        try:
            if _ws_client is not None:
                try:
                    await _ws_client.close()
                except Exception:
                    pass
            _ws_client = await websockets.connect(
                _ws_url, compression=_WS_COMPRESSION
            )
            asyncio.create_task(_receiver(_ws_client))
            if _watch_task is None or _watch_task.done():
                loop = asyncio.get_running_loop()
                _watch_task = loop.create_task(_watch_ws())
            break
        except Exception:  # pragma: no cover - connection errors
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)


async def _watch_ws() -> None:
    """Background task to watch the websocket connection."""
    while _ws_url:
        ws = _ws_client
        if ws is None:
            await reconnect_ws()
            ws = _ws_client
        if ws is None:
            await asyncio.sleep(1.0)
            continue
        try:
            await ws.wait_closed()
        except Exception:
            await asyncio.sleep(0.1)
        if _ws_url:
            await reconnect_ws()


# Automatically connect to an external event bus if configured
_ENV_URL = None


def _reload_bus(cfg) -> None:
    global _ENV_URL
    url = _get_bus_url(cfg)
    if url == _ENV_URL:
        return

    async def _reconnect() -> None:
        if url:
            await reconnect_ws(url)
        else:
            await disconnect_ws()

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


async def send_heartbeat(
    service: str,
    interval: float = 30.0,
    metrics_interval: float | None = None,
) -> None:
    """Publish heartbeat for ``service`` every ``interval`` seconds."""
    if metrics_interval:
        from . import resource_monitor

        resource_monitor.start_monitor(metrics_interval)
    while True:
        publish("heartbeat", {"service": service})
        await asyncio.sleep(interval)

