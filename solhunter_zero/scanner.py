from __future__ import annotations
import os
import requests
import logging
import time
from typing import List, Dict

logger = logging.getLogger(__name__)

BIRDEYE_API = "https://public-api.birdeye.so/defi/tokenlist"  # Example placeholder
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")
HEADERS: Dict[str, str] = {}
if BIRDEYE_API_KEY:
    HEADERS["X-API-KEY"] = BIRDEYE_API_KEY

def scan_tokens() -> List[str]:
    """Scan the Solana network for new tokens ending with 'bonk'."""
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
