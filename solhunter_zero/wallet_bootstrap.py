from __future__ import annotations

import logging
import os
from pathlib import Path

from . import wallet


def ensure_active_keypair(one_click: bool) -> tuple[wallet.KeypairInfo, Path]:
    """Ensure a usable keypair exists and is selected.

    Parameters
    ----------
    one_click:
        When ``True`` informational messages are logged via ``logging`` instead
        of printed, and automatic keypair selection is enabled.

    Returns
    -------
    tuple[wallet.KeypairInfo, Path]
        The :class:`~solhunter_zero.wallet.KeypairInfo` describing the active
        keypair and the path to its JSON file.
    """

    log = logging.getLogger(__name__)
    if one_click and os.getenv("AUTO_SELECT_KEYPAIR") is None:
        os.environ["AUTO_SELECT_KEYPAIR"] = "1"

    def _msg(msg: str) -> None:
        if one_click:
            log.info(msg)
        else:
            print(msg)

    keypair_json = os.environ.get("KEYPAIR_JSON")
    try:
        result = wallet.setup_default_keypair()
    except Exception as exc:  # pragma: no cover - handled interactively
        _msg(f"Failed to set up default keypair: {exc}")
        if keypair_json:
            os.environ.pop("KEYPAIR_JSON", None)
            _msg("Removed KEYPAIR_JSON environment variable.")
        if one_click:
            raise SystemExit(1)
        input("Press Enter to retry without KEYPAIR_JSON or Ctrl+C to abort...")
        result = wallet.setup_default_keypair()

    name, mnemonic_path = result.name, result.mnemonic_path
    keypair_path = Path(wallet.KEYPAIR_DIR) / f"{name}.json"

    if keypair_json:
        _msg("Keypair saved from KEYPAIR_JSON and selected as 'default'.")
        _msg(f"Keypair stored at {keypair_path}.")
    elif mnemonic_path:
        _msg(f"Generated mnemonic and keypair '{name}'.")
        _msg(f"Keypair stored at {keypair_path}.")
        _msg(f"Mnemonic stored at {mnemonic_path}.")
        if not one_click:
            _msg("Please store this mnemonic securely; it will not be shown again.")
    else:
        _msg(f"Using keypair '{name}'.")

    return result, keypair_path
