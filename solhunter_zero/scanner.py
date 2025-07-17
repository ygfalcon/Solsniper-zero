from __future__ import annotations
import requests
import logging
import time
from typing import List

logger = logging.getLogger(__name__)

BIRDEYE_API = "https://public-api.birdeye.so/defi/tokenlist"  # Example placeholder

def scan_tokens() -> List[str]:
    """Scan the Solana network for new tokens ending with 'bonk'."""
    backoff = 1
    max_backoff = 60
    while True:
        try:
            resp = requests.get(BIRDEYE_API, timeout=10)
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
