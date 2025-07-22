import os
from . import order_book_ws
from .arbitrage import DEPTH_SERVICE_SOCKET
from typing import AsyncGenerator, Dict, Any, Optional

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
