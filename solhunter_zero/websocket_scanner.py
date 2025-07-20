"""Scan new token mints via Solana websocket logs."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncGenerator, Iterable

from solana.publickey import PublicKey
from solana.rpc.websocket_api import (
    RpcTransactionLogsFilterMentions,
    connect,
)

from .scanner_onchain import TOKEN_PROGRAM_ID
from .scanner_common import (
    TOKEN_SUFFIX,
    TOKEN_KEYWORDS,
    JUPITER_WS_URL,
    token_matches,
)


logger = logging.getLogger(__name__)

NAME_RE = re.compile(r"name:\s*(\S+)", re.IGNORECASE)
MINT_RE = re.compile(r"mint:\s*(\S+)", re.IGNORECASE)


from .scanner_common import TOKEN_SUFFIX, TOKEN_KEYWORDS, token_matches


async def stream_new_tokens(
    rpc_url: str,
    *,
    suffix: str | None = None,
    keywords: Iterable[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Yield new token mint addresses passing configured filters.

    Parameters
    ----------
    rpc_url:
        Websocket endpoint of a Solana RPC node.
    suffix:
        Token name suffix to filter on. Case-insensitive.
    """

    if not rpc_url:
        raise ValueError("rpc_url is required")

    if suffix is None:
        suffix = TOKEN_SUFFIX
    if keywords is None:
        keywords = TOKEN_KEYWORDS
    suffix = suffix.lower() if suffix else None

    async with connect(rpc_url) as ws:
        await ws.logs_subscribe(
            RpcTransactionLogsFilterMentions(PublicKey(str(TOKEN_PROGRAM_ID))._key)
        )

        while True:
            try:
                msgs = await ws.recv()
            except Exception as exc:  # pragma: no cover - network errors
                logger.error("Websocket error: %s", exc)
                await asyncio.sleep(1)
                continue

            for msg in msgs:
                try:
                    logs = msg.result.value.logs  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover - unexpected format
                    try:
                        logs = msg["result"]["value"]["logs"]  # type: ignore[index]
                    except Exception:
                        continue

                if not any("InitializeMint" in l for l in logs):
                    continue

                name = None
                mint = None
                for log_line in logs:
                    if name is None:
                        m = NAME_RE.search(log_line)
                        if m:
                            name = m.group(1)
                    if mint is None:
                        m = MINT_RE.search(log_line)
                        if m:
                            mint = m.group(1)

                if name and mint and token_matches(mint, name, suffix=suffix, keywords=keywords):
                    yield mint


async def stream_jupiter_tokens(
    url: str = JUPITER_WS_URL,
    *,
    suffix: str | None = None,
    keywords: Iterable[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Yield tokens from the Jupiter aggregator websocket."""

    if suffix is None:
        suffix = TOKEN_SUFFIX
    if keywords is None:
        keywords = TOKEN_KEYWORDS

    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            async for msg in ws:
                try:
                    data = msg.json()
                except Exception:  # pragma: no cover - malformed message
                    continue
                addr = (
                    data.get("address")
                    or data.get("mint")
                    or data.get("id")
                )
                name = data.get("name") or data.get("symbol")
                vol = data.get("volume") or data.get("volume_24h")
                if addr and token_matches(addr, name, vol, suffix=suffix, keywords=keywords):
                    yield addr
