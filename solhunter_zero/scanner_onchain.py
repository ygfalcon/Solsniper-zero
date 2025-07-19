from __future__ import annotations

import logging
import time
from typing import List

from solana.publickey import PublicKey
from solana.rpc.api import Client

logger = logging.getLogger(__name__)

TOKEN_PROGRAM_ID = PublicKey("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")


def scan_tokens_onchain(rpc_url: str) -> List[str]:
    """Query recent token accounts from the blockchain and return mints whose
    names end with ``bonk``.

    Parameters
    ----------
    rpc_url:
        Solana RPC endpoint.
    """
    if not rpc_url:
        raise ValueError("rpc_url is required")

    client = Client(rpc_url)

    backoff = 1
    max_backoff = 60
    attempts = 0
    while True:
        try:
            resp = client.get_program_accounts(
                TOKEN_PROGRAM_ID, encoding="jsonParsed"
            )
            break
        except Exception as exc:  # pragma: no cover - network errors
            attempts += 1
            if attempts >= 5:
                logger.error("On-chain scan failed: %s", exc)
                return []
            logger.warning(
                "RPC error: %s. Sleeping %s seconds before retry", exc, backoff
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    tokens: List[str] = []
    for acc in resp.get("result", []):
        info = (
            acc.get("account", {})
            .get("data", {})
            .get("parsed", {})
            .get("info", {})
        )
        name = info.get("name", "")
        mint = info.get("mint")
        if name and name.lower().endswith("bonk"):
            tokens.append(mint)
    logger.info("Found %d candidate on-chain tokens", len(tokens))
    return tokens
