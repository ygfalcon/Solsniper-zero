from __future__ import annotations

import os
import platform

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


def bootstrap(one_click: bool = False) -> None:
    """Initialize the runtime environment for SolHunter Zero.

    This helper mirrors the setup performed by ``scripts/startup.py`` and can
    be used by entry points that need to guarantee the project is ready to run
    programmatically.
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if not device.MPS_SENTINEL.exists():
            device.ensure_torch_with_metal()
    device.ensure_gpu_env()

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
        ensure_keypair()

    wallet.ensure_default_keypair()
    ensure_route_ffi()
    ensure_depth_service()
