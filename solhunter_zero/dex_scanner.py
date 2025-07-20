from __future__ import annotations

import logging
from typing import List

try:
    from solana.publickey import PublicKey  # type: ignore
except Exception:  # pragma: no cover - fallback when solana not installed
    class PublicKey(str):
        def __new__(cls, value: str):
            return str.__new__(cls, value)

    import types, sys
    mod = types.ModuleType("solana.publickey")
    mod.PublicKey = PublicKey
    sys.modules.setdefault("solana.publickey", mod)

from solana.rpc.api import Client

logger = logging.getLogger(__name__)

DEX_PROGRAM_ID = PublicKey("9xQeWvG816bUx9EPB8YVJprFLaDpbZc81FNtdVUL5J7")


def scan_new_pools(rpc_url: str) -> List[str]:
    """Return BONK-related token mints from recently created pools."""
    if not rpc_url:
        raise ValueError("rpc_url is required")

    client = Client(rpc_url)
    try:
        resp = client.get_program_accounts(DEX_PROGRAM_ID, encoding="jsonParsed")
    except Exception as exc:  # pragma: no cover - network errors
        logger.error("Pool scan failed: %s", exc)
        return []

    tokens: List[str] = []
    for acc in resp.get("result", []):
        info = (
            acc.get("account", {})
            .get("data", {})
            .get("parsed", {})
            .get("info", {})
        )
        for key in ("tokenA", "tokenB"):
            mint = info.get(key, {}).get("mint")
            if mint and mint.lower().endswith("bonk"):
                tokens.append(mint)
    logger.info("Found %d tokens from pools", len(tokens))
    return tokens
