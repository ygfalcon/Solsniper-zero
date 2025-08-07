from __future__ import annotations

import logging
import os
from pathlib import Path

from solhunter_zero.bootstrap_utils import (
    DepsConfig,
    ensure_cargo,
    ensure_deps,
    ensure_venv,
)
from .config_bootstrap import ensure_config as _ensure_config
from . import wallet
from . import env

import solhunter_zero.device as device


def ensure_route_ffi() -> None:
    from .build_utils import ensure_route_ffi as _ensure_route_ffi

    _ensure_route_ffi()


def ensure_depth_service() -> None:
    from .build_utils import ensure_depth_service as _ensure_depth_service

    _ensure_depth_service()


def ensure_config() -> Path:
    """Ensure a configuration file exists and is valid and return its path."""

    return _ensure_config()


def ensure_keypair() -> tuple["wallet.KeypairInfo", Path]:
    """Ensure a usable keypair exists and is selected.

    Returns the :class:`~solhunter_zero.wallet.KeypairInfo` and path to the
    JSON keypair file.
    """

    log = logging.getLogger(__name__)
    one_click = os.getenv("AUTO_SELECT_KEYPAIR") == "1"

    def _msg(msg: str) -> None:
        if one_click:
            log.info(msg)
        else:
            print(msg)

    keypair_json = os.environ.get("KEYPAIR_JSON")
    try:
        result = wallet.setup_default_keypair()
    except Exception as exc:  # pragma: no cover - handled interactively
        print(f"Failed to set up default keypair: {exc}")
        if keypair_json:
            os.environ.pop("KEYPAIR_JSON", None)
            print("Removed KEYPAIR_JSON environment variable.")
        if one_click:
            raise SystemExit(1)
        input(
            "Press Enter to retry without KEYPAIR_JSON or Ctrl+C to abort..."
        )
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


def bootstrap(one_click: bool = False) -> None:
    """Initialize the runtime environment for SolHunter Zero.

    This helper mirrors the setup performed by ``scripts/startup.py`` and can
    be used by entry points that need to guarantee the project is ready to run
    programmatically. It automatically loads the project's ``.env`` file,
    making it self-contained regarding environment setup.
    """
    env.load_env_file(Path(__file__).resolve().parent.parent / ".env")
    device.initialize_gpu()

    if one_click:
        os.environ.setdefault("AUTO_SELECT_KEYPAIR", "1")

    if os.getenv("SOLHUNTER_SKIP_VENV") != "1":
        ensure_venv(None)

    if os.getenv("SOLHUNTER_SKIP_DEPS") != "1":
        cfg = DepsConfig(
            install_optional=os.getenv("SOLHUNTER_INSTALL_OPTIONAL") == "1"
        )
        ensure_deps(cfg)

    if os.getenv("SOLHUNTER_SKIP_SETUP") != "1":
        ensure_config()
        ensure_keypair()

    wallet.ensure_default_keypair()
    ensure_cargo()
