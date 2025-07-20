"""Scan new token mints via Solana websocket logs."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncGenerator

from solana.publickey import PublicKey
from solana.rpc.websocket_api import (
    RpcTransactionLogsFilterMentions,
    connect,
)

from .scanner_onchain import TOKEN_PROGRAM_ID
from .dex_scanner import DEX_PROGRAM_ID


logger = logging.getLogger(__name__)

NAME_RE = re.compile(r"name:\s*(\S+)", re.IGNORECASE)
MINT_RE = re.compile(r"mint:\s*(\S+)", re.IGNORECASE)


POOL_TOKEN_RE = re.compile(r"token[AB]:\s*(\S+)", re.IGNORECASE)


async def stream_new_tokens(
    rpc_url: str, *, suffix: str = "bonk", include_pools: bool = True
) -> AsyncGenerator[str, None]:
    """Yield new token mint addresses or pool tokens matching ``suffix``.

    Parameters
    ----------
    rpc_url:
        Websocket endpoint of a Solana RPC node.
    suffix:
        Token name suffix to filter on. Case-insensitive.
    """

    if not rpc_url:
        raise ValueError("rpc_url is required")

    suffix = suffix.lower()

    async with connect(rpc_url) as ws:
        await ws.logs_subscribe(
            RpcTransactionLogsFilterMentions(PublicKey(str(TOKEN_PROGRAM_ID))._key)
        )
        if include_pools:
            await ws.logs_subscribe(
                RpcTransactionLogsFilterMentions(PublicKey(str(DEX_PROGRAM_ID))._key)
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

                tokens = set()

                if any("InitializeMint" in l for l in logs):
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

                    if name and mint and name.lower().endswith(suffix):
                        tokens.add(mint)

                if include_pools:
                    for log_line in logs:
                        m = POOL_TOKEN_RE.search(log_line)
                        if m:
                            tok = m.group(1)
                            if tok.lower().endswith(suffix):
                                tokens.add(tok)

                for token in tokens:
                    yield token

