"""Token scanning utilities."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import List

from solders.pubkey import Pubkey
from solana.rpc.api import Client

logger = logging.getLogger(__name__)

RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")


def scan_tokens(limit: int = 100) -> List[str]:
    """Return a list of token mint addresses that end with ``bonk``."""

    client = Client(RPC_URL)

    try:
        sig_resp = client.get_signatures_for_address(TOKEN_PROGRAM, limit=limit)
        sigs = sig_resp.value
    except Exception as exc:
        logger.error("RPC error fetching signatures: %s", exc)
        return []

    tokens: List[str] = []
    for info in sigs:
        sig = info.signature
        attempts = 0
        while attempts < 5:
            try:
                tx_resp = client.get_transaction(
                    sig,
                    encoding="jsonParsed",
                    max_supported_transaction_version=0,
                )
                parsed = json.loads(tx_resp.to_json())
                tx = parsed.get("result", {}).get("transaction", {})
                break
            except Exception as exc:
                attempts += 1
                if "429" in str(exc):
                    time.sleep(0.5)
                    continue
                logger.error("RPC error getting tx %s: %s", sig, exc)
                tx = {}
                break
        if not tx:
            continue
        time.sleep(0.2)
        for inst in tx.get("message", {}).get("instructions", []):
            parsed = inst.get("parsed", {})
            if inst.get("program") == "spl-token" and parsed.get("type") in {"initializeMint", "initializeMint2"}:
                mint = parsed.get("info", {}).get("mint")
                if mint and mint.lower().endswith("bonk"):
                    tokens.append(mint)

    unique_tokens = list(dict.fromkeys(tokens))
    logger.info("Found %d candidate tokens", len(unique_tokens))
    return unique_tokens
