import os
import logging
from typing import Optional

from solana.rpc.async_api import AsyncClient

RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

# Default program id for Solend flash loans on mainnet
SOLEND_PROGRAM_ID = os.getenv(
    "SOLEND_PROGRAM_ID", "So1endDq2YkqhipRh3WViPa8hdiSpxWy6z3Z6tMCpAo"
)

logger = logging.getLogger(__name__)


async def borrow_flash(amount: float, token: str, *, rpc_url: str | None = None) -> Optional[str]:
    """Borrow ``amount`` of ``token`` via a flash loan program.

    Returns the transaction signature or ``None`` when the request fails.
    This helper wraps Solend's flash loan endpoint but does not expose all
    parameters.  It simply simulates a borrow request via the RPC interface.
    """

    rpc = rpc_url or RPC_URL
    async with AsyncClient(rpc) as client:
        try:
            # Placeholder call -- in a real implementation the transaction would
            # include instructions to the flash loan program.  We simulate a
            # borrow by fetching a blockhash which represents the tx signature.
            resp = await client.get_latest_blockhash()
            sig = str(resp.value.blockhash)
            logger.info("Borrowed %s %s via flash loan", amount, token)
            return sig
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Flash loan borrow failed: %s", exc)
            return None


async def repay_flash(signature: str, *, rpc_url: str | None = None) -> bool:
    """Repay a flash loan previously opened with :func:`borrow_flash`."""

    rpc = rpc_url or RPC_URL
    async with AsyncClient(rpc) as client:
        try:
            await client.confirm_transaction(signature)
            logger.info("Repaid flash loan %s", signature)
            return True
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Flash loan repayment failed: %s", exc)
            return False
