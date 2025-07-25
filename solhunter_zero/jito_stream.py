from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any

import aiohttp

logger = logging.getLogger(__name__)


async def stream_pending_transactions(
    url: str,
    *,
    auth: str | None = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Yield pending transactions from Jito's searcher websocket.

    Parameters
    ----------
    url:
        WebSocket endpoint of the Jito searcher service.
    auth:
        Optional authentication token.
    """

    headers = {"Authorization": auth} if auth else None

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url, headers=headers) as ws:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                            except Exception:  # pragma: no cover - invalid message
                                continue
                            yield data
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
        except asyncio.CancelledError:
            break
        except Exception as exc:  # pragma: no cover - network errors
            logger.error("Jito stream error: %s", exc)
            await asyncio.sleep(1)
