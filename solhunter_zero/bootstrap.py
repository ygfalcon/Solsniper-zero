"""Runtime bootstrap helpers for SolHunter Zero.

This module prepares the environment before any SolHunter Zero components run.
It ensures PyTorch with Metal support is available prior to configuring the GPU
backend.  Any initialization failure is reported to the user and halts startup
so misconfigured systems do not proceed in a bad state.
"""

from __future__ import annotations

import os
import sys

from scripts.startup import (
    ensure_venv,
    ensure_deps,
    ensure_keypair,
    ensure_route_ffi,
    ensure_depth_service,
)
from .config_bootstrap import ensure_config

import solhunter_zero.device as device


def bootstrap(one_click: bool = False) -> None:
    """Initialize the runtime environment for SolHunter Zero.

    This helper mirrors the setup performed by ``scripts/startup.py`` and can
    be used by entry points that need to guarantee the project is ready to run
    programmatically.
    """
    try:
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

        ensure_route_ffi()
        ensure_depth_service()
    except Exception as exc:  # pragma: no cover - startup errors are surfaced to user
        print(f"[bootstrap] Startup failed: {exc}")
        raise SystemExit(1) from exc
