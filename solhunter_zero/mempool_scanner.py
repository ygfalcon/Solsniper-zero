from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncGenerator, Iterable

from solana.publickey import PublicKey
from solana.rpc.websocket_api import RpcTransactionLogsFilterMentions, connect

from .scanner_onchain import TOKEN_PROGRAM_ID
from .dex_scanner import DEX_PROGRAM_ID
from .scanner_common import TOKEN_SUFFIX, TOKEN_KEYWORDS, token_matches

logger = logging.getLogger(__name__)

NAME_RE = re.compile(r"name:\s*(\S+)", re.IGNORECASE)
MINT_RE = re.compile(r"mint:\s*(\S+)", re.IGNORECASE)
POOL_TOKEN_RE = re.compile(r"token[AB]:\s*([A-Za-z0-9]{32,44})", re.IGNORECASE)

async def stream_mempool_tokens(
    rpc_url: str,
    *,
    suffix: str | None = None,
    keywords: Iterable[str] | None = None,
    include_pools: bool = True,
) -> AsyncGenerator[str, None]:
    """Yield token mints from unconfirmed transactions."""

    if not rpc_url:
        if False:
            yield None
        return

    if suffix is None:
        suffix = TOKEN_SUFFIX
    if keywords is None:
        keywords = TOKEN_KEYWORDS
    suffix = suffix.lower() if suffix else None

    async with connect(rpc_url) as ws:
        await ws.logs_subscribe(
            RpcTransactionLogsFilterMentions(PublicKey(str(TOKEN_PROGRAM_ID))._key),
            commitment="processed",
        )
        if include_pools:
            await ws.logs_subscribe(
                RpcTransactionLogsFilterMentions(PublicKey(str(DEX_PROGRAM_ID))._key),
                commitment="processed",
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
                except Exception:
                    try:
                        logs = msg["result"]["value"]["logs"]  # type: ignore[index]
                    except Exception:
                        continue

                tokens = set()
                if any("InitializeMint" in l for l in logs):
                    name = None
                    mint = None
                    for line in logs:
                        if name is None:
                            m = NAME_RE.search(line)
                            if m:
                                name = m.group(1)
                        if mint is None:
                            m = MINT_RE.search(line)
                            if m:
                                mint = m.group(1)
                    if name and mint and token_matches(mint, name, suffix=suffix, keywords=keywords):
                        tokens.add(mint)

                if include_pools:
                    for line in logs:
                        m = POOL_TOKEN_RE.search(line)
                        if m:
                            tok = m.group(1)
                            if token_matches(tok, None, suffix=suffix, keywords=keywords):
                                tokens.add(tok)

                for tok in tokens:
                    yield tok
