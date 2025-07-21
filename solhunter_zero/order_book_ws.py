from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Cache of latest bid/ask per token
_DEPTH_CACHE: Dict[str, Dict[str, float]] = {}


def snapshot(token: str) -> tuple[float, float]:
    """Return current depth and imbalance for ``token``."""
    data = _DEPTH_CACHE.get(token)
    if not data:
        return 0.0, 0.0
    bids = float(data.get("bids", 0.0))
    asks = float(data.get("asks", 0.0))
    depth = bids + asks
    imbalance = (bids - asks) / depth if depth else 0.0
    return depth, imbalance


async def stream_order_book(
    url: str,
    *,
    rate_limit: float = 0.1,
    max_updates: Optional[int] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Yield depth updates from ``url`` with reconnection and rate limiting."""

    count = 0
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url) as ws:
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        try:
                            data = json.loads(msg.data)
                        except Exception:
                            continue
                        token = data.get("token")
                        bids = float(data.get("bids", 0.0))
                        asks = float(data.get("asks", 0.0))
                        if not token:
                            continue
                        _DEPTH_CACHE[token] = {"bids": bids, "asks": asks}
                        depth, imb = snapshot(token)
                        yield {"token": token, "depth": depth, "imbalance": imb}
                        count += 1
                        if max_updates is not None and count >= max_updates:
                            return
                        if rate_limit > 0:
                            await asyncio.sleep(rate_limit)
        except Exception as exc:  # pragma: no cover - network errors
            logger.error("Order book websocket error: %s", exc)
            await asyncio.sleep(1.0)
