import os
import logging
from typing import Optional, Dict, Any

from solana.rpc.api import Client
from solana.transaction import Transaction
from solders.keypair import Keypair

logger = logging.getLogger(__name__)

# RPC endpoints for submitting transactions.
DEX_BASE_URL = os.getenv("DEX_BASE_URL", "https://api.mainnet-beta.solana.com")
DEX_TESTNET_URL = os.getenv("DEX_TESTNET_URL", "https://api.devnet.solana.com")


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

    endpoint = DEX_TESTNET_URL if testnet else DEX_BASE_URL
    cluster = "devnet" if testnet else "mainnet-beta"
    dex = Client(endpoint)

    if dry_run:
        logger.info(
            "Dry run: would place %s order for %s amount %s at price %s", side, token, amount, price
        )
        return {"dry_run": True, "cluster": cluster}

    tx = Transaction()
    payer = Keypair()  # In real usage load from a wallet

    try:
        resp = dex.send_transaction(tx, payer)
        return resp
    except Exception as exc:  # pragma: no cover - network errors
        logger.error("Order submission failed: %s", exc)
        return None
