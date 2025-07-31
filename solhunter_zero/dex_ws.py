from __future__ import annotations

from .jsonutil import loads
import logging
import os
from typing import AsyncGenerator, Iterable

import aiohttp
from .http import get_session

from .scanner_common import (
    TOKEN_SUFFIX,
    TOKEN_KEYWORDS,
    token_matches,
    PHOENIX_WS_URL,
    METEORA_WS_URL,
)

DEX_LISTING_WS_URL = os.getenv("DEX_LISTING_WS_URL", "")
PHOENIX_DEPTH_WS_URL = os.getenv("PHOENIX_DEPTH_WS_URL", PHOENIX_WS_URL)
METEORA_DEPTH_WS_URL = os.getenv("METEORA_DEPTH_WS_URL", METEORA_WS_URL)

logger = logging.getLogger(__name__)


async def stream_listed_tokens(
    url: str = DEX_LISTING_WS_URL,
    *,
    suffix: str | None = None,
    keywords: Iterable[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Yield token addresses from a DEX or mempool websocket."""

    if suffix is None:
        suffix = TOKEN_SUFFIX
    if keywords is None:
        keywords = TOKEN_KEYWORDS

    if not url:
        return

    session = await get_session()
    async with session.ws_connect(url) as ws:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    data = loads(msg.data)
                except Exception:  # pragma: no cover - invalid message
                    continue
                addr = data.get("address") or data.get("mint") or data.get("id")
                name = data.get("name") or data.get("symbol")
                vol = data.get("volume") or data.get("volume_24h")
                if addr and token_matches(addr, name, vol, suffix=suffix, keywords=keywords):
                    yield addr
