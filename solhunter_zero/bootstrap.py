from __future__ import annotations

import logging
import os
from pathlib import Path

from solhunter_zero.bootstrap_utils import (
    ensure_cargo,
    ensure_deps,
    ensure_venv,
)
from .config_bootstrap import ensure_config as _ensure_config
from . import wallet
from . import env
from .wallet_bootstrap import ensure_active_keypair

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
        ensure_deps(
            install_optional=os.getenv("SOLHUNTER_INSTALL_OPTIONAL") == "1"
        )

    if os.getenv("SOLHUNTER_SKIP_SETUP") != "1":
        ensure_config()
        ensure_active_keypair(one_click)

    wallet.ensure_default_keypair()
    ensure_cargo()
