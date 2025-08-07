from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from solhunter_zero.bootstrap_utils import (
    ensure_deps,
    ensure_depth_service,
    ensure_route_ffi,
    ensure_venv,
)
from scripts.startup import ensure_keypair
from .config_bootstrap import ensure_config
from . import wallet

import solhunter_zero.device as device


def bootstrap(one_click: bool = False) -> Dict[str, Any]:
    """Initialize the runtime environment for SolHunter Zero.

    This helper mirrors the setup performed by ``scripts/startup.py`` and can
    be used by entry points that need to guarantee the project is ready to run
    programmatically. When ``one_click`` is ``True`` the helper also performs
    the configuration and keypair setup just like ``scripts/startup.main`` and
    returns details about the created artifacts for diagnostics.
    """
    device.ensure_gpu_env()

    if one_click:
        os.environ.setdefault("AUTO_SELECT_KEYPAIR", "1")

    if os.getenv("SOLHUNTER_SKIP_VENV") != "1":
        ensure_venv(None)

    if os.getenv("SOLHUNTER_SKIP_DEPS") != "1":
        ensure_deps(
            install_optional=os.getenv("SOLHUNTER_INSTALL_OPTIONAL") == "1"
        )

    info: Dict[str, Any] = {}
    if one_click and os.getenv("SOLHUNTER_SKIP_SETUP") != "1":
        cfg_path = ensure_config()
        kp_info, keypair_path = ensure_keypair()
        info.update(
            {
                "config_path": Path(cfg_path),
                "keypair_path": keypair_path,
                "mnemonic_path": kp_info.mnemonic_path,
                "active_keypair": kp_info.name,
            }
        )

    wallet.ensure_default_keypair()
    ensure_route_ffi()
    ensure_depth_service()

    return info
