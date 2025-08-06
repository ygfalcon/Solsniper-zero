from __future__ import annotations

import os

from pathlib import Path

from scripts.startup import (
    ensure_venv,
    ensure_deps,
    ensure_config,
    ensure_keypair,
    ensure_route_ffi,
    ensure_depth_service,
)


def bootstrap(one_click: bool = False) -> tuple[str | None, Path | None, Path | None]:
    """Initialize the runtime environment for SolHunter Zero.

    This helper mirrors the setup performed by ``scripts/startup.py`` and can
    be used by entry points that need to guarantee the project is ready to run
    programmatically.  The tuple returned mirrors the result of
    :func:`scripts.startup.ensure_keypair` and contains the active keypair name,
    the path to the keypair file and, when generated, the path to the mnemonic
    file.
    """
    if one_click:
        os.environ.setdefault("AUTO_SELECT_KEYPAIR", "1")

    if os.getenv("SOLHUNTER_SKIP_VENV") != "1":
        ensure_venv(None)

    if os.getenv("SOLHUNTER_SKIP_DEPS") != "1":
        ensure_deps(install_optional=os.getenv("SOLHUNTER_INSTALL_OPTIONAL") == "1")

    key_info: tuple[str | None, Path | None, Path | None] = (None, None, None)
    if os.getenv("SOLHUNTER_SKIP_SETUP") != "1":
        ensure_config()
        key_info = ensure_keypair()

    ensure_route_ffi()
    ensure_depth_service()
    return key_info
