import os
import json
import mmap
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
