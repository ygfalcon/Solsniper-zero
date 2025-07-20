import os
from solana.rpc.api import Client
from solana.rpc.async_api import AsyncClient

RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
RPC_TESTNET_URL = os.getenv("SOLANA_TESTNET_RPC_URL", "https://api.devnet.solana.com")

LAMPORTS_PER_SOL = 1_000_000_000


def _extract_lamports(resp: object) -> int:
    """Return lamports per signature from an RPC response."""
    try:
        value = resp["value"]  # type: ignore[index]
    except Exception:
        value = getattr(resp, "value", {})
    if isinstance(value, dict):
        calc = value.get("feeCalculator") or value.get("fee_calculator") or {}
        lamports = calc.get("lamportsPerSignature") or calc.get("lamports_per_signature")
        if isinstance(lamports, (int, float)):
            return int(lamports)
    try:
        return int(value.fee_calculator.lamports_per_signature)  # type: ignore[attr-defined]
    except Exception:
        return 0


def get_current_fee(testnet: bool = False) -> float:
    """Return current fee per signature in SOL."""
    client = Client(RPC_TESTNET_URL if testnet else RPC_URL)
    resp = client.get_fees()
    lamports = _extract_lamports(resp)
    return lamports / LAMPORTS_PER_SOL


async def get_current_fee_async(testnet: bool = False) -> float:
    """Asynchronously return current fee per signature in SOL."""
    async with AsyncClient(RPC_TESTNET_URL if testnet else RPC_URL) as client:
        resp = await client.get_fees()
    lamports = _extract_lamports(resp)
    return lamports / LAMPORTS_PER_SOL
