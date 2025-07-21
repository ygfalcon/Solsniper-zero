import os
import base64
import logging
from typing import Optional, Dict, Any
import asyncio
import json

import requests
import aiohttp

IPC_SOCKET = os.getenv("DEPTH_SERVICE_SOCKET", "/tmp/depth_service.sock")
USE_RUST_EXEC = os.getenv("USE_RUST_EXEC", "0").lower() in {"1", "true", "yes"}

from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.api import Client
from solana.rpc.async_api import AsyncClient

from .gas import get_current_fee_async


class OrderPlacementError(Exception):
    """Raised when an order cannot be placed."""


logger = logging.getLogger(__name__)

# Using Jupiter Aggregator REST API for token swaps.
DEX_BASE_URL = os.getenv("DEX_BASE_URL", "https://quote-api.jup.ag")
DEX_TESTNET_URL = os.getenv("DEX_TESTNET_URL", "https://quote-api.jup.ag")
ORCA_DEX_URL = os.getenv("ORCA_DEX_URL", DEX_BASE_URL)
RAYDIUM_DEX_URL = os.getenv("RAYDIUM_DEX_URL", DEX_BASE_URL)
SWAP_PATH = "/v6/swap"

RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
RPC_TESTNET_URL = os.getenv("SOLANA_TESTNET_RPC_URL", "https://api.devnet.solana.com")


def _sign_transaction(tx_b64: str, keypair: Keypair) -> VersionedTransaction:
    tx = VersionedTransaction.from_bytes(base64.b64decode(tx_b64))
    sig = keypair.sign_message(bytes(tx.message))
    return VersionedTransaction.populate(tx.message, [sig] + tx.signatures[1:])


async def _place_order_ipc(
    token: str,
    side: str,
    amount: float,
    price: float,
    *,
    testnet: bool = False,
    dry_run: bool = False,
    socket_path: str = IPC_SOCKET,
) -> Optional[Dict[str, Any]]:
    if dry_run:
        logger.info(
            "Dry run IPC order %s %s %s at %s", side, token, amount, price
        )
        return {"dry_run": True}
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)
        payload = {
            "cmd": "order",
            "token": token,
            "side": side,
            "amount": amount,
            "price": price,
            "testnet": testnet,
        }
        writer.write(json.dumps(payload).encode())
        await writer.drain()
        data = await reader.read()
        writer.close()
        await writer.wait_closed()
        return json.loads(data.decode()) if data else None
    except Exception as exc:
        logger.error("IPC order submission failed: %s", exc)
        return None


def place_order(
    token: str,
    side: str,
    amount: float,
    price: float,
    *,
    testnet: bool = False,
    dry_run: bool = False,
    keypair: Keypair | None = None,
    base_url: str | None = None,
) -> Optional[Dict[str, Any]]:
    """Submit an order to the Jupiter swap API and broadcast the transaction."""

    if base_url is None:
        base_url = DEX_TESTNET_URL if testnet else DEX_BASE_URL
    url = f"{base_url}{SWAP_PATH}"

    payload = {
        "token": token,
        "side": side,
        "amount": amount,
        "price": price,
        "cluster": "devnet" if testnet else "mainnet-beta",
    }

    if dry_run:
        logger.info(
            "Dry run: would place %s order for %s amount %s at price %s",
            side,
            token,
            amount,
            price,
        )
        return {"dry_run": True, **payload}

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        tx_b64 = data.get("swapTransaction")
        if not tx_b64 or keypair is None:
            return data
        tx = _sign_transaction(tx_b64, keypair)
        rpc = Client(RPC_TESTNET_URL if testnet else RPC_URL)
        result = rpc.send_raw_transaction(bytes(tx))
        data["signature"] = str(result.value)
        return data
    except requests.HTTPError as exc:
        data = getattr(exc.response, "text", "") if getattr(exc, "response", None) else ""
        status = exc.response.status_code if getattr(exc, "response", None) else "no-response"
        logger.error("Order failed with status %s: %s", status, data)
        raise OrderPlacementError(f"HTTP {status}: {data}") from exc
    except requests.RequestException as exc:
        logger.error("Order submission failed: %s", exc)
        return None


async def place_order_async(
    token: str,
    side: str,
    amount: float,
    price: float,
    *,
    testnet: bool = False,
    dry_run: bool = False,
    keypair: Keypair | None = None,
    base_url: str | None = None,
) -> Optional[Dict[str, Any]]:
    """Asynchronously submit an order and broadcast the transaction."""

    if USE_RUST_EXEC:
        return await _place_order_ipc(
            token,
            side,
            amount,
            price,
            testnet=testnet,
            dry_run=dry_run,
        )

    if base_url is None:
        base_url = DEX_TESTNET_URL if testnet else DEX_BASE_URL
    url = f"{base_url}{SWAP_PATH}"

    fee = await get_current_fee_async(testnet=testnet)
    trade_amount = max(0.0, amount - fee)

    payload = {
        "token": token,
        "side": side,
        "amount": trade_amount,
        "price": price,
        "cluster": "devnet" if testnet else "mainnet-beta",
    }

    if dry_run:
        logger.info(
            "Dry run: would place %s order for %s amount %s at price %s",
            side,
            token,
            amount,
            price,
        )
        return {"dry_run": True, **payload}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()

        tx_b64 = data.get("swapTransaction")
        if not tx_b64 or keypair is None:
            return data

        tx = _sign_transaction(tx_b64, keypair)
        async with AsyncClient(RPC_TESTNET_URL if testnet else RPC_URL) as client:
            result = await client.send_raw_transaction(bytes(tx))
        data["signature"] = str(result.value)
        return data
    except aiohttp.ClientError as exc:
        logger.error("Order submission failed: %s", exc)
        return None
