from __future__ import annotations
import requests
import logging
from typing import List

logger = logging.getLogger(__name__)

BIRDEYE_API = "https://public-api.birdeye.so/defi/tokenlist"  # Example placeholder

def scan_tokens() -> List[str]:
    """Scan the Solana network for new tokens ending with 'bonk'."""
    try:
        resp = requests.get(BIRDEYE_API, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        tokens = [t['address'] for t in data.get('data', []) if t['address'].lower().endswith('bonk')]
        logger.info("Found %d candidate tokens", len(tokens))
        return tokens
    except Exception as e:
        logger.error("Scan failed: %s", e)
        return []
