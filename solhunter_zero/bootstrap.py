from __future__ import annotations

import os

from scripts.startup import (
    ensure_venv,
    ensure_deps,
    ensure_config,
    ensure_keypair,
    ensure_route_ffi,
    ensure_depth_service,
)

from . import device


def bootstrap(one_click: bool = False) -> None:
    """Initialize the runtime environment for SolHunter Zero.

    This helper mirrors the setup performed by ``scripts/startup.py`` and can
    be used by entry points that need to guarantee the project is ready to run
    programmatically.
    """
    if one_click:
        os.environ.setdefault("AUTO_SELECT_KEYPAIR", "1")

    if os.getenv("SOLHUNTER_SKIP_VENV") != "1":
        ensure_venv(None)

    if os.getenv("SOLHUNTER_SKIP_DEPS") != "1":
        ensure_deps(install_optional=os.getenv("SOLHUNTER_INSTALL_OPTIONAL") == "1")

    device.ensure_gpu_env()

    if os.getenv("SOLHUNTER_SKIP_SETUP") != "1":
        ensure_config()
        ensure_keypair()

    ensure_route_ffi()
    ensure_depth_service()
