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
import mmap

try:  # optional compression libraries
    import lz4.frame
    _HAS_LZ4 = True
except Exception:  # pragma: no cover - optional dependency
    lz4 = None
    _HAS_LZ4 = False

try:
    import zstandard as zstd
    _HAS_ZSTD = True
    _ZSTD_COMPRESSOR = zstd.ZstdCompressor()
    _ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()
except Exception:  # pragma: no cover - optional dependency
    zstd = None
    _HAS_ZSTD = False
    _ZSTD_COMPRESSOR = None
    _ZSTD_DECOMPRESSOR = None
from contextlib import contextmanager
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, Generator, List, Set

try:
    import msgpack
except Exception:  # pragma: no cover - optional dependency
    msgpack = None

try:  # optional redis / nats support
    import redis.asyncio as aioredis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    aioredis = None

try:
    import nats  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nats = None

from asyncio import Queue


from .schemas import validate_message, to_dict
from . import event_pb2 as pb

_PB_MAP = {
    "action_executed": getattr(pb, "ActionExecuted", None),
    "weights_updated": getattr(pb, "WeightsUpdated", None),
    "rl_weights": getattr(pb, "RLWeights", None),
    "rl_checkpoint": getattr(pb, "RLCheckpoint", None),
    "portfolio_updated": getattr(pb, "PortfolioUpdated", None),
    "depth_update": getattr(pb, "DepthUpdate", None),
    "depth_diff": getattr(pb, "DepthDiff", None),
    "depth_service_status": getattr(pb, "DepthServiceStatus", None),
    "heartbeat": getattr(pb, "Heartbeat", None),
    "trade_logged": getattr(pb, "TradeLogged", None),
    "rl_metrics": getattr(pb, "RLMetrics", None),
    "system_metrics": getattr(pb, "SystemMetrics", None),
    "price_update": getattr(pb, "PriceUpdate", None),
    "config_updated": getattr(pb, "ConfigUpdated", None),
    "pending_swap": getattr(pb, "PendingSwap", None),
    "remote_system_metrics": getattr(pb, "RemoteSystemMetrics", None),
    "risk_metrics": getattr(pb, "RiskMetrics", None),
    "risk_updated": getattr(pb, "RiskUpdated", None),
    "system_metrics_combined": getattr(pb, "SystemMetricsCombined", None),
    "token_discovered": getattr(pb, "TokenDiscovered", None),
    "memory_sync_request": getattr(pb, "MemorySyncRequest", None),
    "memory_sync_response": getattr(pb, "MemorySyncResponse", None),
}

# compress protobuf messages when broadcasting if enabled
_COMPRESS_EVENTS = os.getenv("COMPRESS_EVENTS")
COMPRESS_EVENTS = _COMPRESS_EVENTS not in (None, "", "0")

# chosen compression algorithm for protobuf events
_EVENT_COMPRESSION = os.getenv("EVENT_COMPRESSION")
_USE_ZLIB_EVENTS = os.getenv("USE_ZLIB_EVENTS")
EVENT_COMPRESSION_THRESHOLD = int(
    os.getenv("EVENT_COMPRESSION_THRESHOLD", "512") or 512
)
if _EVENT_COMPRESSION is None:
    if COMPRESS_EVENTS:
        if _USE_ZLIB_EVENTS:
            EVENT_COMPRESSION = "zlib"
        elif _HAS_ZSTD:
            EVENT_COMPRESSION = "zstd"
        else:
            EVENT_COMPRESSION = "zlib"
    else:
        EVENT_COMPRESSION = None
else:
    comp = _EVENT_COMPRESSION.lower()
    if comp in {"", "none", "0"}:
        EVENT_COMPRESSION = None
    else:
        EVENT_COMPRESSION = comp

# optional mmap ring buffer path for outgoing protobuf frames
_EVENT_BUS_MMAP = os.getenv("EVENT_BUS_MMAP")
_EVENT_BUS_MMAP_SIZE = int(os.getenv("EVENT_BUS_MMAP_SIZE", str(1 << 20)) or (1 << 20))
_MMAP: mmap.mmap | None = None


def _maybe_decompress(data: bytes) -> bytes:
    """Return decompressed ``data`` if it appears to be compressed."""
    if len(data) > 4 and _HAS_ZSTD and data.startswith(b"\x28\xb5\x2f\xfd"):
        try:
            return _ZSTD_DECOMPRESSOR.decompress(data)
        except Exception:
            return data
    if len(data) > 4 and _HAS_LZ4 and data.startswith(b"\x04\x22\x4d\x18"):
        try:
            return lz4.frame.decompress(data)
        except Exception:
            return data
    if len(data) > 2 and data[0] == 0x78:
        try:
            return zlib.decompress(data)
        except Exception:
            return data
    return data


def _compress_event(data: bytes) -> bytes:
    """Compress ``data`` using the selected algorithm if any."""
    if len(data) < EVENT_COMPRESSION_THRESHOLD:
        return data
    if EVENT_COMPRESSION in {"zstd", "zstandard"} and _HAS_ZSTD:
        return _ZSTD_COMPRESSOR.compress(data)
    if EVENT_COMPRESSION == "lz4" and _HAS_LZ4:
        return lz4.frame.compress(data)
    if EVENT_COMPRESSION == "zlib":
        return zlib.compress(data)
    return data


def _dumps(obj: Any) -> bytes:
    """Serialize ``obj`` to JSON bytes using ``orjson`` when available."""
    if _USE_ORJSON:
        return json.dumps(obj)
    return json.dumps(obj).encode()


def _loads(data: Any) -> Any:
    """Deserialize JSON ``data`` using ``orjson`` when available."""
    if _USE_ORJSON:
        if isinstance(data, str):
            data = data.encode()
        return json.loads(data)
    if isinstance(data, bytes):
        data = data.decode()
    return json.loads(data)


def open_mmap(path: str | None = None, size: int | None = None) -> mmap.mmap | None:
    """Open or return the configured event mmap ring buffer."""
    global _MMAP
    if _MMAP is not None and not getattr(_MMAP, "closed", False):
        return _MMAP
    if path is None:
        path = _EVENT_BUS_MMAP
    if not path:
        return None
    if size is None:
        size = _EVENT_BUS_MMAP_SIZE
    try:
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT)
        try:
            if os.path.getsize(path) < size:
                os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        if mm.size() >= 4 and int.from_bytes(mm[:4], "little") == 0:
            mm[:4] = (4).to_bytes(4, "little")
        _MMAP = mm
    except Exception:
        _MMAP = None
    return _MMAP


def close_mmap() -> None:
    """Close the event mmap if open."""
    global _MMAP
    if _MMAP is not None:
        try:
            _MMAP.close()
        except Exception:
            pass
    _MMAP = None


def _mmap_write(data: bytes) -> None:
    mm = open_mmap()
    if mm is None or len(data) == 0:
        return
    size = mm.size()
    if size < 8:
        return
    pos = int.from_bytes(mm[:4], "little") or 4
    if pos < 4 or pos + 4 + len(data) > size:
        pos = 4
    if len(data) > size - 8:
        data = data[: size - 8]
    mm[pos:pos+4] = len(data).to_bytes(4, "little")
    mm[pos+4:pos+4+len(data)] = data
    pos += 4 + len(data)
    if pos >= size:
        pos = 4
    mm[:4] = pos.to_bytes(4, "little")
    try:
        mm.flush()
    except Exception:
        pass


def _get_bus_url(cfg=None):
    from .config import get_event_bus_url
    return get_event_bus_url(cfg)

try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    websockets = None

# compression algorithm for websockets
_WS_COMPRESSION: str | None = os.getenv("EVENT_BUS_COMPRESSION")
if _WS_COMPRESSION:
    comp = _WS_COMPRESSION.lower()
    if comp in {"", "none", "0"}:
        _WS_COMPRESSION = None

# magic header for batched binary websocket messages
_BATCH_MAGIC = b"EBAT"
_MP_BATCH_MAGIC = b"EBMP"


def _pack_batch(msgs: List[Any]) -> Any:
    """Return a single frame containing all ``msgs``."""
    if not msgs:
        return b""
    if _WS_COMPRESSION and msgpack is not None:
        return _MP_BATCH_MAGIC + msgpack.packb(msgs)
    first = msgs[0]
    if isinstance(first, bytes):
        parts = [len(m).to_bytes(4, "big") + m for m in msgs]
        return _BATCH_MAGIC + len(msgs).to_bytes(4, "big") + b"".join(parts)
    return _dumps(msgs)


def _unpack_batch(data: Any) -> List[Any] | None:
    """Return list of messages if ``data`` is a batched frame."""
    if isinstance(data, bytes):
        if len(data) >= len(_MP_BATCH_MAGIC) and data.startswith(_MP_BATCH_MAGIC) and msgpack is not None:
            try:
                obj = msgpack.unpackb(data[len(_MP_BATCH_MAGIC):])
            except Exception:
                return None
            if isinstance(obj, list):
                return list(obj)
            return None
        if len(data) >= len(_BATCH_MAGIC) + 4 and data.startswith(_BATCH_MAGIC):
            off = len(_BATCH_MAGIC)
            count = int.from_bytes(data[off:off + 4], "big")
            off += 4
            msgs = []
            for _ in range(count):
                if off + 4 > len(data):
                    break
                ln = int.from_bytes(data[off:off + 4], "big")
                off += 4
                msgs.append(data[off:off + ln])
                off += ln
            if len(msgs) == count:
                return msgs
        return None
    if isinstance(data, str):
        try:
            obj = _loads(data)
        except Exception:
            return None
        if isinstance(obj, list):
            return obj
    return None

# mapping of topic -> list of handlers
_subscribers: Dict[str, List[Callable[[Any], Awaitable[None] | None]]] = defaultdict(list)

# websocket related globals
_ws_clients: Set[Any] = set()  # clients connected to our server
_peer_clients: Dict[str, Any] = {}  # outbound connections to peers
_peer_urls: Set[str] = set()
_watch_tasks: Dict[str, Any] = {}
_ws_server = None
_flush_task = None
_outgoing_queue: Queue | None = None
_flush_task = None
_outgoing_queue: Queue | None = None

# message broker globals
_BROKER_URL: str | None = None
_BROKER_TYPE: str | None = None
_BROKER_CONN: Any | None = None
_BROKER_TASK: Any | None = None
_BROKER_CHANNEL: str = os.getenv("BROKER_CHANNEL", "solhunter-events")


def _encode_event(topic: str, payload: Any) -> Any:
    cls = _PB_MAP.get(topic)
    if cls is None:
        event = pb.Event(topic=topic)
        data = event.SerializeToString()
        return _compress_event(data)

    if topic == "action_executed":
        event = pb.Event(
            topic=topic,
            action_executed=pb.ActionExecuted(
                action_json=_dumps(payload.action).decode(),
                result_json=_dumps(payload.result).decode(),
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
    elif topic in {"depth_update", "depth_diff"}:
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
        if topic == "depth_update":
            event = pb.Event(topic=topic, depth_update=pb.DepthUpdate(entries=entries))
        else:
            event = pb.Event(topic=topic, depth_diff=pb.DepthDiff(entries=entries))
        data = event.SerializeToString()
        return _compress_event(data)
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
    elif topic == "config_updated":
        event = pb.Event(
            topic=topic,
            config_updated=pb.ConfigUpdated(
                config_json=_dumps(to_dict(payload)).decode(),
            ),
        )
    elif topic == "pending_swap":
        data = to_dict(payload)
        event = pb.Event(
            topic=topic,
            pending_swap=pb.PendingSwap(
                token=str(data.get("token", "")),
                address=str(data.get("address", "")),
                size=float(data.get("size", 0.0)),
                slippage=float(data.get("slippage", 0.0)),
            ),
        )
    elif topic == "remote_system_metrics":
        data = to_dict(payload)
        event = pb.Event(
            topic=topic,
            remote_system_metrics=pb.RemoteSystemMetrics(
                cpu=float(data.get("cpu", 0.0)),
                memory=float(data.get("memory", 0.0)),
            ),
        )
    elif topic == "risk_metrics":
        data = to_dict(payload)
        event = pb.Event(
            topic=topic,
            risk_metrics=pb.RiskMetrics(
                covariance=float(data.get("covariance", 0.0)),
                portfolio_cvar=float(data.get("portfolio_cvar", 0.0)),
                portfolio_evar=float(data.get("portfolio_evar", 0.0)),
                correlation=float(data.get("correlation", 0.0)),
                cov_matrix=[pb.DoubleList(values=[float(x) for x in row]) for row in data.get("cov_matrix", [])],
                corr_matrix=[pb.DoubleList(values=[float(x) for x in row]) for row in data.get("corr_matrix", [])],
            ),
        )
    elif topic == "risk_updated":
        data = to_dict(payload)
        event = pb.Event(
            topic=topic,
            risk_updated=pb.RiskUpdated(
                multiplier=float(data.get("multiplier", 0.0)),
            ),
        )
    elif topic == "system_metrics_combined":
        data = to_dict(payload)
        event = pb.Event(
            topic=topic,
            system_metrics_combined=pb.SystemMetricsCombined(
                cpu=float(data.get("cpu", 0.0)),
                memory=float(data.get("memory", 0.0)),
            ),
        )
    elif topic == "token_discovered":
        data = payload
        if not isinstance(data, (list, tuple)):
            data = to_dict(data)
        event = pb.Event(
            topic=topic,
            token_discovered=pb.TokenDiscovered(tokens=[str(t) for t in data]),
        )
    elif topic == "memory_sync_request":
        data = to_dict(payload)
        event = pb.Event(
            topic=topic,
            memory_sync_request=pb.MemorySyncRequest(last_id=int(data.get("last_id", 0))),
        )
    elif topic == "memory_sync_response":
        data = to_dict(payload)
        trades = [
            pb.TradeLogged(
                token=str(t.get("token", "")),
                direction=str(t.get("direction", "")),
                amount=float(t.get("amount", 0.0)),
                price=float(t.get("price", 0.0)),
                reason=str(t.get("reason", "")),
                context=str(t.get("context", "")),
                emotion=str(t.get("emotion", "")),
                simulation_id=int(t.get("simulation_id", 0) or 0),
                uuid=str(t.get("uuid", "")),
                trade_id=int(t.get("trade_id", 0) or 0),
            )
            for t in (data.get("trades") or [])
        ]
        event = pb.Event(
            topic=topic,
            memory_sync_response=pb.MemorySyncResponse(trades=trades, index=data.get("index") or b""),
        )
    data = event.SerializeToString()
    return _compress_event(data)

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
    if field in {"depth_update", "depth_diff"}:
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
    if field == "config_updated":
        return _loads(msg.config_json)
    if field == "pending_swap":
        return {
            "token": msg.token,
            "address": msg.address,
            "size": msg.size,
            "slippage": msg.slippage,
        }
    if field == "remote_system_metrics":
        return {"cpu": msg.cpu, "memory": msg.memory}
    if field == "risk_metrics":
        cov = [list(row.values) for row in msg.cov_matrix]
        corr = [list(row.values) for row in msg.corr_matrix]
        return {
            "covariance": msg.covariance,
            "portfolio_cvar": msg.portfolio_cvar,
            "portfolio_evar": msg.portfolio_evar,
            "correlation": msg.correlation,
            "cov_matrix": cov,
            "corr_matrix": corr,
        }
    if field == "risk_updated":
        return {"multiplier": msg.multiplier}
    if field == "system_metrics_combined":
        return {"cpu": msg.cpu, "memory": msg.memory}
    if field == "token_discovered":
        return list(msg.tokens)
    if field == "memory_sync_request":
        return {"last_id": msg.last_id}
    if field == "memory_sync_response":
        trades = [
            {
                "token": t.token,
                "direction": t.direction,
                "amount": t.amount,
                "price": t.price,
                "reason": t.reason,
                "context": t.context,
                "emotion": t.emotion,
                "simulation_id": t.simulation_id,
                "uuid": t.uuid,
                "trade_id": t.trade_id,
            }
            for t in msg.trades
        ]
        return {"trades": trades, "index": bytes(msg.index)}
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

    msg: Any | None = None
    if (websockets or _BROKER_TYPE or _EVENT_BUS_MMAP) and _broadcast:
        msg = _encode_event(topic, payload)
    if _EVENT_BUS_MMAP and _broadcast and isinstance(msg, (bytes, bytearray)):
        _mmap_write(bytes(msg))
    if websockets and _broadcast:
        assert msg is not None
        if loop:
            if _outgoing_queue is not None:
                _outgoing_queue.put_nowait(msg)
            else:
                loop.create_task(broadcast_ws(msg))
        else:
            asyncio.run(broadcast_ws(msg))
    if _BROKER_TYPE and _broadcast:
        assert msg is not None
        if loop:
            loop.create_task(_broker_send(msg))
        else:
            asyncio.run(_broker_send(msg))


async def broadcast_ws(
    message: Any,
    *,
    to_clients: bool = True,
    to_server: bool = True,
) -> None:
    """Send ``message`` to websocket peers."""
    if isinstance(message, list):
        message = _pack_batch(message)
    if to_clients:
        clients = list(_ws_clients)
        coros = [ws.send(message) for ws in clients]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for ws, res in zip(clients, results):
            if isinstance(res, Exception):  # pragma: no cover - network errors
                _ws_clients.discard(ws)
    if to_server and _peer_clients:
        peers = list(_peer_clients.items())
        coros = [ws.send(message) for _, ws in peers]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for (url, ws), res in zip(peers, results):
            if isinstance(res, Exception):  # pragma: no cover - connection issues
                asyncio.create_task(reconnect_ws(url))


async def _flush_outgoing() -> None:
    """Background task to flush queued websocket messages."""
    global _outgoing_queue
    q = _outgoing_queue
    if q is None:
        return
    while True:
        # wait for at least one message
        msg = await q.get()
        msgs: list[Any] = [msg]
        # drain the queue until empty so we batch dispatch
        while True:
            try:
                msgs.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
        if len(msgs) == 1:
            await broadcast_ws(msgs[0])
        else:
            await broadcast_ws(msgs)


async def _broker_send(message: bytes) -> None:
    """Publish ``message`` to the configured message broker."""
    if _BROKER_TYPE == "redis" and _BROKER_CONN is not None:
        try:
            await _BROKER_CONN.publish(_BROKER_CHANNEL, message)
        except Exception:  # pragma: no cover - connection issues
            pass
    elif _BROKER_TYPE == "nats" and _BROKER_CONN is not None:
        try:
            await _BROKER_CONN.publish(_BROKER_CHANNEL, message)
        except Exception:  # pragma: no cover - connection issues
            pass


async def _receiver(ws) -> None:
    """Receive messages from ``ws`` and publish them."""
    try:
        async for msg in ws:
            try:
                msgs = _unpack_batch(msg) or [msg]
                for single in msgs:
                    if isinstance(single, bytes):
                        data = _maybe_decompress(single)
                        ev = pb.Event()
                        ev.ParseFromString(data)
                        payload = _decode_payload(ev)
                        publish(ev.topic, payload, _broadcast=False)
                    else:
                        data = _loads(single)
                        topic = data.get("topic")
                        payload = data.get("payload")
                        publish(topic, payload, _broadcast=False)
                await broadcast_ws(msg, to_server=False)
            except Exception:  # pragma: no cover - malformed msg
                continue
    finally:
        for url, peer in list(_peer_clients.items()):
            if ws is peer and url in _peer_urls:
                asyncio.create_task(reconnect_ws(url))
                break


async def _redis_listener(pubsub) -> None:
    """Listen to Redis pub/sub messages."""
    try:
        async for msg in pubsub.listen():
            if msg.get("type") != "message":
                continue
            data = msg.get("data")
            if isinstance(data, memoryview):
                data = bytes(data)
            ev = pb.Event()
            ev.ParseFromString(_maybe_decompress(data))
            publish(ev.topic, _decode_payload(ev), _broadcast=False)
    except asyncio.CancelledError:  # pragma: no cover - shutdown
        pass


async def _connect_nats(url: str):
    nc = nats.NATS()
    await nc.connect(servers=[url])

    async def _cb(msg):
        ev = pb.Event()
        ev.ParseFromString(_maybe_decompress(msg.data))
        publish(ev.topic, _decode_payload(ev), _broadcast=False)

    await nc.subscribe(_BROKER_CHANNEL, cb=_cb)
    return nc


async def connect_broker(url: str) -> None:
    """Connect to a Redis or NATS message broker."""
    global _BROKER_URL, _BROKER_CONN, _BROKER_TASK, _BROKER_TYPE
    if url.startswith("redis://") or url.startswith("rediss://"):
        if aioredis is None:
            raise RuntimeError("redis package required for redis broker")
        conn = aioredis.from_url(url)
        pubsub = conn.pubsub()
        await pubsub.subscribe(_BROKER_CHANNEL)
        _BROKER_TASK = asyncio.create_task(_redis_listener(pubsub))
        _BROKER_TYPE = "redis"
        _BROKER_CONN = conn
    elif url.startswith("nats://"):
        if nats is None:
            raise RuntimeError("nats-py package required for nats broker")
        _BROKER_CONN = await _connect_nats(url)
        _BROKER_TYPE = "nats"
        _BROKER_TASK = None
    else:
        raise ValueError(f"unsupported broker url: {url}")
    _BROKER_URL = url


async def disconnect_broker() -> None:
    """Disconnect from the active message broker."""
    global _BROKER_URL, _BROKER_CONN, _BROKER_TASK, _BROKER_TYPE
    task = _BROKER_TASK
    if task is not None:
        task.cancel()
        _BROKER_TASK = None
    if _BROKER_TYPE == "redis" and _BROKER_CONN is not None:
        try:
            await _BROKER_CONN.close()
        except Exception:
            pass
    elif _BROKER_TYPE == "nats" and _BROKER_CONN is not None:
        try:
            await _BROKER_CONN.close()
        except Exception:
            pass
    _BROKER_URL = None
    _BROKER_CONN = None
    _BROKER_TYPE = None


async def start_ws_server(host: str = "localhost", port: int = 8765):
    """Start websocket server broadcasting published events."""
    if not websockets:
        raise RuntimeError("websockets library required")

    async def handler(ws):
        _ws_clients.add(ws)
        try:
            async for msg in ws:
                try:
                    msgs = _unpack_batch(msg) or [msg]
                    for single in msgs:
                        if isinstance(single, bytes):
                            data = _maybe_decompress(single)
                            ev = pb.Event()
                            ev.ParseFromString(data)
                            publish(ev.topic, _decode_payload(ev), _broadcast=False)
                        else:
                            data = _loads(single)
                            publish(data.get("topic"), data.get("payload"), _broadcast=False)
                except Exception:  # pragma: no cover - malformed message
                    continue
        finally:
            _ws_clients.discard(ws)

    global _ws_server, _flush_task, _outgoing_queue
    if _outgoing_queue is None:
        _outgoing_queue = Queue()
    if _flush_task is None or _flush_task.done():
        loop = asyncio.get_running_loop()
        _flush_task = loop.create_task(_flush_outgoing())

    _ws_server = await websockets.serve(
        handler, host, port, compression=_WS_COMPRESSION
    )
    return _ws_server


async def stop_ws_server() -> None:
    """Stop the running websocket server and close client connections."""
    global _ws_server, _flush_task, _outgoing_queue
    if _ws_server is not None:
        _ws_server.close()
        await _ws_server.wait_closed()
        _ws_server = None
    if _flush_task is not None:
        _flush_task.cancel()
        _flush_task = None
    _outgoing_queue = None
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

    if _WS_COMPRESSION is None:
        ws = await websockets.connect(url)
    else:
        ws = await websockets.connect(url, compression=_WS_COMPRESSION)
    try:
        from . import resource_monitor

        resource_monitor.start_monitor()
    except Exception:  # pragma: no cover - optional psutil failure
        pass
    _peer_clients[url] = ws
    _peer_urls.add(url)
    asyncio.create_task(_receiver(ws))
    task = _watch_tasks.get(url)
    if task is None or task.done():
        loop = asyncio.get_running_loop()
        _watch_tasks[url] = loop.create_task(_watch_ws(url))
    return ws


async def disconnect_ws() -> None:
    """Close websocket connection opened via ``connect_ws``."""
    global _peer_clients, _peer_urls, _watch_tasks
    for ws in list(_peer_clients.values()):
        try:
            await ws.close()
        except Exception:
            pass
    _peer_clients.clear()
    _peer_urls.clear()
    for t in _watch_tasks.values():
        t.cancel()
    _watch_tasks.clear()
    try:
        from . import resource_monitor

        resource_monitor.stop_monitor()
    except Exception:
        pass


async def reconnect_ws(url: str | None = None) -> None:
    """(Re)connect the websocket client with exponential backoff."""
    if url:
        _peer_urls.add(url)
    if not websockets:
        return
    urls = [url] if url else list(_peer_urls)
    for u in urls:
        backoff = 1.0
        max_backoff = 30
        while True:
            try:
                ws = _peer_clients.get(u)
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                if _WS_COMPRESSION is None:
                    ws = await websockets.connect(u)
                else:
                    ws = await websockets.connect(u, compression=_WS_COMPRESSION)
                try:
                    from . import resource_monitor

                    resource_monitor.start_monitor()
                except Exception:  # pragma: no cover - optional psutil failure
                    pass
                _peer_clients[u] = ws
                asyncio.create_task(_receiver(ws))
                task = _watch_tasks.get(u)
                if task is None or task.done():
                    loop = asyncio.get_running_loop()
                    _watch_tasks[u] = loop.create_task(_watch_ws(u))
                break
            except Exception:  # pragma: no cover - connection errors
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)


async def _watch_ws(url: str) -> None:
    """Background task to watch the websocket connection for ``url``."""
    while url in _peer_urls:
        ws = _peer_clients.get(url)
        if ws is None:
            await reconnect_ws(url)
            ws = _peer_clients.get(url)
        if ws is None:
            await asyncio.sleep(1.0)
            continue
        try:
            await ws.wait_closed()
        except Exception:
            await asyncio.sleep(0.1)
        if url in _peer_urls:
            await reconnect_ws(url)


# Automatically connect to external event bus peers if configured
_ENV_PEERS: Set[str] = set()


def _reload_bus(cfg) -> None:
    global _ENV_PEERS
    urls = set(
        u.strip()
        for u in os.getenv("EVENT_BUS_PEERS", "").split(",")
        if u.strip()
    )
    single = _get_bus_url(cfg)
    if single:
        urls.add(single)
    if urls == _ENV_PEERS:
        return

    async def _reconnect() -> None:
        await disconnect_ws()
        for u in urls:
            await connect_ws(u)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop:
        loop.create_task(_reconnect())
    else:
        asyncio.run(_reconnect())
    _ENV_PEERS = urls


subscription("config_updated", _reload_bus).__enter__()


# ---------------------------------------------------------------------------
#  Message broker integration
# ---------------------------------------------------------------------------

_ENV_BROKER: str | None = None


def _get_broker_url(cfg=None):
    from .config import get_broker_url
    return get_broker_url(cfg)


def _reload_broker(cfg) -> None:
    global _ENV_BROKER
    url = _get_broker_url(cfg)
    if url == _ENV_BROKER:
        return

    async def _reconnect() -> None:
        await disconnect_broker()
        if url:
            await connect_broker(url)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop:
        loop.create_task(_reconnect())
    else:
        asyncio.run(_reconnect())
    _ENV_BROKER = url


subscription("config_updated", _reload_broker).__enter__()

# initialize mmap on import if configured
open_mmap()


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

