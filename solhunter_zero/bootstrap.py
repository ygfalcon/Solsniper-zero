from __future__ import annotations

import os

from solhunter_zero.bootstrap_utils import (
    ensure_deps,
    ensure_depth_service,
    ensure_route_ffi,
    ensure_venv,
)
from solhunter_zero.bootstrap_checks import (
    check_internet,
    ensure_rpc,
    ensure_endpoints,
    check_disk_space,
    log_startup_failure,
)
from scripts.startup import ensure_keypair
from .config_bootstrap import ensure_config
from .config import load_config, validate_config
from . import wallet

import solhunter_zero.device as device


def bootstrap(one_click: bool = False) -> None:
    """Initialize the runtime environment for SolHunter Zero.

    This helper mirrors the setup performed by ``scripts/startup.py`` and can
    be used by entry points that need to guarantee the project is ready to run
    programmatically.
    """
    device.ensure_gpu_env()

    if one_click:
        os.environ.setdefault("AUTO_SELECT_KEYPAIR", "1")

    if os.getenv("SOLHUNTER_SKIP_INTERNET") != "1":
        check_internet()

    if os.getenv("SOLHUNTER_SKIP_RPC") != "1":
        ensure_rpc()

    if os.getenv("SOLHUNTER_SKIP_ENDPOINTS") != "1":
        try:
            cfg = validate_config(load_config())
        except Exception as exc:  # pragma: no cover - config failure
            log_startup_failure(f"Failed to load configuration: {exc}")
            raise SystemExit(1) from exc
        ensure_endpoints(cfg)

    if os.getenv("SOLHUNTER_SKIP_DISK") != "1":
        check_disk_space(1 << 30)

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
