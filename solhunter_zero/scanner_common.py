import os
import logging
import asyncio
from typing import Dict, List, Optional
from pathlib import Path

from .scanner_onchain import scan_tokens_onchain

logger = logging.getLogger(__name__)

BIRDEYE_API = "https://public-api.birdeye.so/defi/tokenlist"
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL")

HEADERS: Dict[str, str] = {}
if BIRDEYE_API_KEY:
    HEADERS["X-API-KEY"] = BIRDEYE_API_KEY
else:
    logger.warning(
        "BIRDEYE_API_KEY not set. Falling back to on-chain scanning by default"
    )

OFFLINE_TOKENS = ["offlinebonk1", "offlinebonk2"]


def load_tokens_from_file(path: str) -> List[str]:
    """Return token addresses listed one per line in *path*."""
    tokens: List[str] = []
    for line in Path(path).read_text().splitlines():
        tok = line.strip()
        if tok and not tok.startswith("#"):
            tokens.append(tok)
    logger.info("Loaded %d tokens from %s", len(tokens), path)
    return tokens


def parse_birdeye_tokens(data: dict) -> List[str]:
    tokens = [
        t["address"]
        for t in data.get("data", [])
        if t["address"].lower().endswith("bonk")
    ]
    logger.info("Found %d candidate tokens", len(tokens))
    return tokens


def offline_or_onchain(offline: bool, token_file: str | None = None) -> Optional[List[str]]:
    if token_file:
        return load_tokens_from_file(token_file)
    if offline:
        logger.info("Offline mode enabled, returning static tokens")
        return OFFLINE_TOKENS

    if not BIRDEYE_API_KEY:
        logger.info("No BirdEye API key set, scanning on-chain")
        return scan_tokens_onchain(SOLANA_RPC_URL)

    return None


async def offline_or_onchain_async(offline: bool, token_file: str | None = None) -> Optional[List[str]]:
    if token_file:
        return load_tokens_from_file(token_file)
    if offline:
        logger.info("Offline mode enabled, returning static tokens")
        return OFFLINE_TOKENS

    if not BIRDEYE_API_KEY:
        logger.info("No BirdEye API key set, scanning on-chain")
        return await asyncio.to_thread(scan_tokens_onchain, SOLANA_RPC_URL)

    return None
