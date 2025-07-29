import os
import json
import mmap
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional, Tuple

import aiohttp

from .event_bus import publish, subscription
from .config import get_depth_ws_addr

from . import order_book_ws

DEPTH_SERVICE_SOCKET = os.getenv("DEPTH_SERVICE_SOCKET", "/tmp/depth_service.sock")

MMAP_PATH = os.getenv("DEPTH_MMAP_PATH", "/tmp/depth_service.mmap")
DEPTH_WS_ADDR, DEPTH_WS_PORT = get_depth_ws_addr()


def _reload_depth(cfg) -> None:
    global DEPTH_WS_ADDR, DEPTH_WS_PORT
    DEPTH_WS_ADDR, DEPTH_WS_PORT = get_depth_ws_addr(cfg)


subscription("config_updated", _reload_depth).__enter__()

async def stream_depth(
    token: str,
    *,
    rate_limit: float = 0.1,
    max_updates: Optional[int] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream aggregated depth and mempool rate from the Rust service."""
    url = f"ipc://{DEPTH_SERVICE_SOCKET}?{token}"
    async for data in order_book_ws.stream_order_book(
        url, rate_limit=rate_limit, max_updates=max_updates
    ):
        yield data


async def stream_depth_ws(
    token: str,
    *,
    rate_limit: float = 0.1,
    max_updates: Optional[int] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream depth updates from the WebSocket server with mmap fallback."""
    url = f"ws://{DEPTH_WS_ADDR}:{DEPTH_WS_PORT}"
    count = 0
    was_connected = False
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url) as ws:
                    publish(
                        "depth_service_status",
                        {"status": "reconnected" if was_connected else "connected"},
                    )
                    was_connected = True
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        try:
                            data = json.loads(msg.data)
                        except Exception:
                            continue
                        entry = data.get(token)
                        if not entry:
                            continue
                        bids = float(entry.get("bids", 0.0)) if "bids" in entry else 0.0
                        asks = float(entry.get("asks", 0.0)) if "asks" in entry else 0.0
                        rate = float(entry.get("tx_rate", 0.0))
                        for v in entry.values():
                            if isinstance(v, dict):
                                bids += float(v.get("bids", 0.0))
                                asks += float(v.get("asks", 0.0))
                        depth = bids + asks
                        imb = (bids - asks) / depth if depth else 0.0
                        yield {
                            "token": token,
                            "depth": depth,
                            "imbalance": imb,
                            "tx_rate": rate,
                        }
                        count += 1
                        if max_updates is not None and count >= max_updates:
                            publish("depth_service_status", {"status": "disconnected"})
                            return
                        if rate_limit > 0:
                            await asyncio.sleep(rate_limit)
        except Exception:
            if was_connected:
                publish("depth_service_status", {"status": "disconnected"})
                was_connected = False
            venues, rate = snapshot(token)
            bids = sum(float(v.get("bids", 0.0)) for v in venues.values())
            asks = sum(float(v.get("asks", 0.0)) for v in venues.values())
            depth = bids + asks
            imb = (bids - asks) / depth if depth else 0.0
            yield {
                "token": token,
                "depth": depth,
                "imbalance": imb,
                "tx_rate": rate,
            }
            count += 1
            if max_updates is not None and count >= max_updates:
                return
            if rate_limit > 0:
                await asyncio.sleep(rate_limit)
        else:
            if was_connected:
                publish("depth_service_status", {"status": "disconnected"})
                was_connected = False


async def listen_depth_ws(*, max_updates: Optional[int] = None) -> None:
    """Connect to the depth websocket and publish updates."""

    url = f"ws://{DEPTH_WS_ADDR}:{DEPTH_WS_PORT}"
    count = 0
    was_connected = False
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url) as ws:
                    publish(
                        "depth_service_status",
                        {"status": "reconnected" if was_connected else "connected"},
                    )
                    was_connected = True
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        try:
                            data = json.loads(msg.data)
                        except Exception:
                            continue
                        publish("depth_update", data)
                        count += 1
                        if max_updates is not None and count >= max_updates:
                            publish("depth_service_status", {"status": "disconnected"})
                            return
        except Exception:
            if was_connected:
                publish("depth_service_status", {"status": "disconnected"})
                was_connected = False
            await asyncio.sleep(1.0)
        else:
            if was_connected:
                publish("depth_service_status", {"status": "disconnected"})
                was_connected = False


def snapshot(token: str) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Return order book depth per venue and mempool rate."""
    try:
        with open(MMAP_PATH, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                raw = bytes(m).rstrip(b"\x00")
                if not raw:
                    return {}, 0.0
                data = json.loads(raw.decode())
                entry = data.get(token)
                if not entry:
                    return {}, 0.0
                rate = float(entry.get("tx_rate", 0.0))
                venues = {}
                for dex, info in entry.items():
                    if dex == "tx_rate" or not isinstance(info, dict):
                        continue
                    venues[dex] = {
                        "bids": float(info.get("bids", 0.0)),
                        "asks": float(info.get("asks", 0.0)),
                    }
                return venues, rate
    except Exception:
        return {}, 0.0


async def depth_feed(*, rate_limit: float = 0.5, max_updates: Optional[int] = None) -> None:
    """Publish aggregated depth snapshots via the event bus."""

    count = 0
    while True:
        data: Dict[str, Any] = {}
        try:
            with open(MMAP_PATH, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                    raw = bytes(m).rstrip(b"\x00")
                    if raw:
                        data = json.loads(raw.decode())
        except Exception:
            data = {}

        publish("depth_update", data)
        count += 1
        if max_updates is not None and count >= max_updates:
            return
        if rate_limit > 0:
            await asyncio.sleep(rate_limit)


async def submit_signed_tx(
    tx_b64: str,
    *,
    socket_path: str = DEPTH_SERVICE_SOCKET,
    timeout: float | None = None,
) -> Optional[str]:
    """Forward a pre-signed transaction to the Rust service."""

    reader, writer = await asyncio.open_unix_connection(socket_path)
    payload = {"cmd": "signed_tx", "tx": tx_b64}
    writer.write(json.dumps(payload).encode())
    await writer.drain()
    if timeout is not None:
        data = await asyncio.wait_for(reader.read(), timeout)
    else:
        data = await reader.read()
    writer.close()
    await writer.wait_closed()
    if not data:
        return None
    try:
        resp = json.loads(data.decode())
    except Exception:
        return None
    return resp.get("signature")


async def prepare_signed_tx(
    msg_b64: str,
    *,
    priority_fee: int | None = None,
    socket_path: str = DEPTH_SERVICE_SOCKET,
    timeout: float | None = None,
) -> Optional[str]:
    """Request a signed transaction from a template message."""

    reader, writer = await asyncio.open_unix_connection(socket_path)
    payload: Dict[str, Any] = {"cmd": "prepare", "msg": msg_b64}
    if priority_fee is not None:
        payload["priority_fee"] = int(priority_fee)
    writer.write(json.dumps(payload).encode())
    await writer.drain()
    if timeout is not None:
        data = await asyncio.wait_for(reader.read(), timeout)
    else:
        data = await reader.read()
    writer.close()
    await writer.wait_closed()
    if not data:
        return None
    try:
        resp = json.loads(data.decode())
    except Exception:
        return None
    return resp.get("tx")


async def submit_tx_batch(
    txs: list[str],
    *,
    socket_path: str = DEPTH_SERVICE_SOCKET,
    timeout: float | None = None,
) -> list[str] | None:
    """Submit multiple pre-signed transactions at once."""

    reader, writer = await asyncio.open_unix_connection(socket_path)
    payload = {"cmd": "batch", "txs": txs}
    writer.write(json.dumps(payload).encode())
    await writer.drain()
    if timeout is not None:
        data = await asyncio.wait_for(reader.read(), timeout)
    else:
        data = await reader.read()
    writer.close()
    await writer.wait_closed()
    if not data:
        return None
    try:
        resp = json.loads(data.decode())
    except Exception:
        return None
    if isinstance(resp, list):
        return [str(s) for s in resp]
    return None


async def submit_raw_tx(
    tx_b64: str,
    *,
    priority_rpc: list[str] | None = None,
    priority_fee: int | None = None,
    socket_path: str = DEPTH_SERVICE_SOCKET,
    timeout: float | None = None,
) -> Optional[str]:
    """Submit a transaction with optional priority RPC endpoints."""

    reader, writer = await asyncio.open_unix_connection(socket_path)
    payload: Dict[str, Any] = {"cmd": "raw_tx", "tx": tx_b64}
    if priority_rpc:
        payload["priority_rpc"] = list(priority_rpc)
    if priority_fee is not None:
        payload["priority_fee"] = int(priority_fee)
    writer.write(json.dumps(payload).encode())
    await writer.drain()
    if timeout is not None:
        data = await asyncio.wait_for(reader.read(), timeout)
    else:
        data = await reader.read()
    writer.close()
    await writer.wait_closed()
    if not data:
        return None
    try:
        resp = json.loads(data.decode())
    except Exception:
        return None
    return resp.get("signature")


async def auto_exec(
    token: str,
    threshold: float,
    txs: list[str],
    *,
    socket_path: str = DEPTH_SERVICE_SOCKET,
    timeout: float | None = None,
) -> bool:
    """Register auto-execution triggers with the Rust service."""

    reader, writer = await asyncio.open_unix_connection(socket_path)
    payload: Dict[str, Any] = {
        "cmd": "auto_exec",
        "token": token,
        "threshold": threshold,
        "txs": list(txs),
    }
    writer.write(json.dumps(payload).encode())
    await writer.drain()
    if timeout is not None:
        data = await asyncio.wait_for(reader.read(), timeout)
    else:
        data = await reader.read()
    writer.close()
    await writer.wait_closed()
    if not data:
        return False
    try:
        resp = json.loads(data.decode())
    except Exception:
        return False
    return bool(resp.get("ok"))


async def best_route(
    token: str,
    amount: float,
    *,
    socket_path: str = DEPTH_SERVICE_SOCKET,
    timeout: float | None = None,
    max_hops: int | None = None,
) -> tuple[list[str], float, float] | None:
    """Return the optimal trading path from the Rust service."""

    reader, writer = await asyncio.open_unix_connection(socket_path)
    payload: Dict[str, Any] = {"cmd": "route", "token": token, "amount": amount}
    if max_hops is not None:
        payload["max_hops"] = int(max_hops)
    writer.write(json.dumps(payload).encode())
    await writer.drain()
    if timeout is not None:
        data = await asyncio.wait_for(reader.read(), timeout)
    else:
        data = await reader.read()
    writer.close()
    await writer.wait_closed()
    if not data:
        return None
    try:
        resp = json.loads(data.decode())
        path = [str(p) for p in resp.get("path", [])]
        profit = float(resp.get("profit", 0.0))
        slippage = float(resp.get("slippage", 0.0))
        return path, profit, slippage
    except Exception:
        return None
