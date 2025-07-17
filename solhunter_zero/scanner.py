from __future__ import annotations
import os
import logging
import time
from typing import List, Dict

import requests
from solana.rpc.api import Client

logger = logging.getLogger(__name__)

BIRDEYE_API = "https://public-api.birdeye.so/defi/tokenlist"  # Example placeholder
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")
HEADERS: Dict[str, str] = {}
if BIRDEYE_API_KEY:
    HEADERS["X-API-KEY"] = BIRDEYE_API_KEY

SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

def scan_tokens() -> List[str]:
    """Scan the Solana network for new tokens ending with 'bonk'."""
    if HEADERS:
        backoff = 1
        max_backoff = 60
        while True:
            try:
                resp = requests.get(BIRDEYE_API, headers=HEADERS, timeout=10)
                if resp.status_code == 429:
                    logger.warning("Rate limited (429). Sleeping %s seconds", backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
                    continue
                resp.raise_for_status()
                data = resp.json()
                tokens = [
                    t['address']
                    for t in data.get('data', [])
                    if t['address'].lower().endswith('bonk')
                ]
                logger.info("Found %d candidate tokens", len(tokens))
                backoff = 1
                return tokens
            except requests.RequestException as e:
                logger.error("Scan failed: %s", e)
                return []
    else:
        return scan_tokens_onchain()


def scan_tokens_onchain(limit: int = 100) -> List[str]:
    """Fallback scanner using direct RPC queries when BirdEye is unavailable."""
    client = Client(SOLANA_RPC_URL)
    try:
        resp = client.get_program_accounts(TOKEN_PROGRAM_ID, encoding="jsonParsed")
        accounts = resp.get("result", [])[:limit]
        tokens = [
            acc["pubkey"]
            for acc in accounts
            if acc["pubkey"].lower().endswith("bonk")
        ]
        logger.info("Found %d candidate tokens via RPC", len(tokens))
        return tokens
    except Exception as e:  # broad catch to cover RPC errors
        logger.error("On-chain scan failed: %s", e)
        return []
