from __future__ import annotations

import logging
import time
from typing import List


try:
    from solana.publickey import PublicKey  # type: ignore
except Exception:  # pragma: no cover - fallback when solana lacks PublicKey
    class PublicKey(str):
        """Minimal stand-in for ``solana.publickey.PublicKey``."""

        def __new__(cls, value: str):
            return str.__new__(cls, value)

    import types, sys
    mod = types.ModuleType("solana.publickey")
    mod.PublicKey = PublicKey
    sys.modules.setdefault("solana.publickey", mod)
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


def fetch_mempool_tx_rate(token: str, rpc_url: str, limit: int = 20) -> float:
    """Return approximate mempool transaction rate for ``token``."""

    if not rpc_url:
        raise ValueError("rpc_url is required")

    client = Client(rpc_url)
    try:
        resp = client.get_signatures_for_address(PublicKey(token), limit=limit)
        entries = resp.get("result", [])
        times = [e.get("blockTime") for e in entries if e.get("blockTime")]
        if len(times) >= 2:
            duration = max(times) - min(times)
            if duration > 0:
                return float(len(times)) / float(duration)
        return float(len(times))
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch mempool rate for %s: %s", token, exc)
        return 0.0


def fetch_whale_wallet_activity(
    token: str, rpc_url: str, threshold: float = 1_000_000.0
) -> float:
    """Return fraction of liquidity held by large accounts."""

    if not rpc_url:
        raise ValueError("rpc_url is required")

    client = Client(rpc_url)
    try:
        resp = client.get_token_largest_accounts(PublicKey(token))
        accounts = resp.get("result", {}).get("value", [])
        total = 0.0
        whales = 0.0
        for acc in accounts:
            val = acc.get("uiAmount", acc.get("amount", 0))
            try:
                bal = float(val)
            except Exception:
                bal = 0.0
            total += bal
            if bal >= threshold:
                whales += bal
        return whales / total if total else 0.0
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch whale activity for %s: %s", token, exc)
        return 0.0
