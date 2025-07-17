import os
import logging
import json
import base64
from typing import Optional, Dict, Any

import requests
from solders.keypair import Keypair
from solana.rpc.api import Client
from solders.transaction import Transaction

logger = logging.getLogger(__name__)

# Using Jupiter Aggregator REST API for token swaps.
DEX_BASE_URL = os.getenv("DEX_BASE_URL", "https://quote-api.jup.ag")
DEX_TESTNET_URL = os.getenv("DEX_TESTNET_URL", "https://quote-api.jup.ag")
SWAP_PATH = "/v6/swap"

# RPC endpoints for submitting signed transactions
RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
RPC_TESTNET_URL = os.getenv("SOLANA_TESTNET_RPC_URL", "https://api.devnet.solana.com")

# Default path of the keypair used to sign transactions
KEYPAIR_PATH = os.getenv(
    "SOLANA_KEYPAIR", os.path.expanduser("~/.config/solana/id.json")
)


def load_keypair(path: str = KEYPAIR_PATH) -> Keypair:
    """Load a Solana keypair from the given JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Keypair.from_secret_key(bytes(data))


def place_order(
    token: str,
    side: str,
    amount: float,
    price: float,
    *,
    testnet: bool = False,
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """Submit an order to the DEX API.

    Parameters
    ----------
    token: str
        Token address.
    side: str
        "buy" or "sell".
    amount: float
        Amount to trade.
    price: float
        Desired price.
    testnet: bool
        Use the testnet endpoint if ``True``.
    dry_run: bool
        If ``True``, do not send any network requests.
    """

    base_url = DEX_TESTNET_URL if testnet else DEX_BASE_URL
    rpc_url = RPC_TESTNET_URL if testnet else RPC_URL
    url = f"{base_url}{SWAP_PATH}"

    try:
        keypair = load_keypair()
    except Exception as exc:  # pragma: no cover - cannot load keypair
        logger.error("Failed to load keypair: %s", exc)
        return None

    # Jupiter requires the target cluster explicitly and the user's public key.
    payload = {
        "token": token,
        "side": side,
        "amount": amount,
        "price": price,
        "cluster": "devnet" if testnet else "mainnet-beta",
        "userPublicKey": str(keypair.public_key),
    }

    if dry_run:
        logger.info(
            "Dry run: would place %s order for %s amount %s at price %s", side, token, amount, price
        )
        return {"dry_run": True, **payload}

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        raw_tx = base64.b64decode(resp.json()["swapTransaction"])
        tx = Transaction.from_bytes(raw_tx)
        tx.sign(keypair)
        client = Client(rpc_url)
        return client.send_transaction(tx, keypair)
    except (requests.RequestException, Exception) as exc:
        logger.error("Order submission failed: %s", exc)
        return None
