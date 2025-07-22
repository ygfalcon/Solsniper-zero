import os
import json
import mmap
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional, Tuple

from . import order_book_ws

DEPTH_SERVICE_SOCKET = os.getenv("DEPTH_SERVICE_SOCKET", "/tmp/depth_service.sock")

MMAP_PATH = os.getenv("DEPTH_MMAP_PATH", "/tmp/depth_service.mmap")

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
    socket_path: str = DEPTH_SERVICE_SOCKET,
    timeout: float | None = None,
) -> Optional[str]:
    """Submit a transaction with optional priority RPC endpoints."""

    reader, writer = await asyncio.open_unix_connection(socket_path)
    payload: Dict[str, Any] = {"cmd": "raw_tx", "tx": tx_b64}
    if priority_rpc:
        payload["priority_rpc"] = list(priority_rpc)
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
