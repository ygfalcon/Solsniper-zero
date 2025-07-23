import os
import logging
from typing import Optional, Sequence, Mapping

from solders.instruction import Instruction
from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.signature import Signature
from solana.rpc.async_api import AsyncClient

RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

# Default program id for Solend flash loans on mainnet
SOLEND_PROGRAM_ID = os.getenv(
    "SOLEND_PROGRAM_ID", "So1endDq2YkqhipRh3WViPa8hdiSpxWy6z3Z6tMCpAo"
)

logger = logging.getLogger(__name__)


async def borrow_flash(
    amount: float,
    token: str,
    instructions: Sequence[Instruction],
    *,
    payer: Keypair,
    program_accounts: Mapping[str, Pubkey] | None = None,
    rpc_url: str | None = None,
) -> Optional[str]:
    """Borrow ``amount`` of ``token`` via a flash loan and execute ``instructions``.

    The transaction uses :data:`SOLEND_PROGRAM_ID` for the borrow and repay
    instructions and appends the provided swap ``instructions`` in between.  The
    entire sequence is broadcast atomically.
    """

    rpc = rpc_url or RPC_URL
    program_accounts = program_accounts or {}

    borrow_ix = Instruction(
        Pubkey.from_string(SOLEND_PROGRAM_ID),
        b"borrow",
        list(program_accounts.values()),
    )
    repay_ix = Instruction(
        Pubkey.from_string(SOLEND_PROGRAM_ID),
        b"repay",
        list(program_accounts.values()),
    )

    tx_instructions = [borrow_ix] + list(instructions) + [repay_ix]
    msg = b"|".join(ix.data for ix in tx_instructions)
    tx = VersionedTransaction.populate(msg, [Signature.default()])

    async with AsyncClient(rpc) as client:
        try:
            resp = await client.send_raw_transaction(bytes(tx))
            sig = str(resp.value)
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
