from __future__ import annotations

import logging
from typing import List

try:
    from solana.publickey import PublicKey  # type: ignore
except Exception:  # pragma: no cover - older solana versions
    from solders.pubkey import Pubkey as PublicKey  # type: ignore
from solana.rpc.api import Client

logger = logging.getLogger(__name__)

if hasattr(PublicKey, "from_string"):
    TOKEN_PROGRAM_ID = PublicKey.from_string(
        "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    )
else:  # pragma: no cover - older API
    TOKEN_PROGRAM_ID = PublicKey(
        "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"  # type: ignore[arg-type]
    )


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
    try:
        resp = client.get_program_accounts(TOKEN_PROGRAM_ID, encoding="jsonParsed")
    except Exception as exc:  # pragma: no cover - network errors
        logger.error("On-chain scan failed: %s", exc)
        return []

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
