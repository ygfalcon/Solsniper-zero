import os
import logging
from typing import Optional, Dict, Any

import requests


class OrderPlacementError(Exception):
    """Raised when an order cannot be placed."""

logger = logging.getLogger(__name__)

# Using Jupiter Aggregator REST API for token swaps.
DEX_BASE_URL = os.getenv("DEX_BASE_URL", "https://quote-api.jup.ag")
DEX_TESTNET_URL = os.getenv("DEX_TESTNET_URL", "https://quote-api.jup.ag")
SWAP_PATH = "/v6/swap"


def place_order(
    token: str,
    side: str,
    amount: float,
    price: float,
    *,
    testnet: bool = False,
    dry_run: bool = False,
    keypair=None,
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
    url = f"{base_url}{SWAP_PATH}"

    # Jupiter requires the target cluster explicitly.
    payload = {
        "token": token,
        "side": side,
        "amount": amount,
        "price": price,
        "cluster": "devnet" if testnet else "mainnet-beta",
    }

    if keypair is not None:
        try:
            import base64
            import json as _json

            message = _json.dumps(payload, sort_keys=True).encode()
            signature = keypair.sign_message(message)
            payload["signature"] = base64.b64encode(bytes(signature)).decode()
        except Exception as exc:  # pragma: no cover - signing errors
            logger.error("Signing failed: %s", exc)

    if dry_run:
        logger.info(
            "Dry run: would place %s order for %s amount %s at price %s", side, token, amount, price
        )
        return {"dry_run": True, **payload}

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        data = getattr(exc.response, "text", "") if getattr(exc, "response", None) else ""
        status = exc.response.status_code if getattr(exc, "response", None) else "no-response"
        logger.error("Order failed with status %s: %s", status, data)
        raise OrderPlacementError(f"HTTP {status}: {data}") from exc
    except requests.RequestException as exc:
        logger.error("Order submission failed: %s", exc)
        raise OrderPlacementError(str(exc)) from exc
