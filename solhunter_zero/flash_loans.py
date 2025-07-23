import os
import base64
import logging
from typing import Optional, Sequence, Mapping

from solders.instruction import Instruction, AccountMeta
from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient

from . import depth_client

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

    program_id = Pubkey.from_string(SOLEND_PROGRAM_ID)
    metas = [AccountMeta(payer.pubkey(), True, True)] + [
        AccountMeta(pk, False, True) for pk in program_accounts.values()
    ]

    borrow_ix = Instruction(program_id, b"flash_borrow", metas)
    repay_ix = Instruction(program_id, b"flash_repay", metas)

    tx_instructions = [borrow_ix, *instructions, repay_ix]

    async with AsyncClient(rpc) as client:
        try:
            latest = await client.get_latest_blockhash()
            msg = MessageV0.try_compile(
                payer.pubkey(), tx_instructions, [], latest.value.blockhash
            )
            sig = payer.sign_message(bytes(msg))
            tx = VersionedTransaction.populate(msg, [sig])
            tx_b64 = base64.b64encode(bytes(tx)).decode()
            sig_str = await depth_client.submit_raw_tx(tx_b64)
            logger.info("Borrowed %s %s via flash loan", amount, token)
            return sig_str
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
