from __future__ import annotations

"""Utilities for bootstrapping the SolHunter Zero runtime."""

import os
import sys
from pathlib import Path

from . import env, wallet
from .device import initialize_gpu
from .logging_utils import log_startup
from .bootstrap_utils import ensure_deps, ensure_venv
from .config_bootstrap import ensure_config

__all__ = ["bootstrap_all"]


def _log_path(msg: str, path: Path | None) -> None:
    if path:
        log_startup(f"{msg}: {path}")


def bootstrap_all(*, install_optional: bool = False, one_click: bool = False) -> tuple[Path | None, wallet.KeypairInfo | None, Path | None]:
    """Run full bootstrap sequence for SolHunter Zero.

    The process performs the following steps:

    * load environment variables from ``.env``
    * initialise GPU support
    * create a virtual environment when necessary
    * install dependencies
    * ensure configuration and default keypair exist
    * write details to ``startup.log``

    Parameters
    ----------
    install_optional:
        When ``True`` optional dependencies defined in :mod:`scripts.deps` are
        installed.
    one_click:
        Enables non-interactive keypair setup by setting
        ``AUTO_SELECT_KEYPAIR``.

    Returns
    -------
    tuple
        ``(config_path, keypair_info, keypair_path)`` where any element may be
        ``None`` when the corresponding setup step was skipped.
    """

    root = Path(__file__).resolve().parent.parent
    env.load_env_file(root / ".env")
    initialize_gpu()
    log_startup(f"Virtual environment: {sys.prefix}")

    if one_click:
        os.environ.setdefault("AUTO_SELECT_KEYPAIR", "1")

    if os.getenv("SOLHUNTER_SKIP_VENV") != "1":
        ensure_venv(None)

    if os.getenv("SOLHUNTER_SKIP_DEPS") != "1":
        ensure_deps(install_optional=install_optional)

    config_path: Path | None = None
    keypair_path: Path | None = None
    info: wallet.KeypairInfo | None = None

    if os.getenv("SOLHUNTER_SKIP_SETUP") != "1":
        config_path = ensure_config()
        info = wallet.setup_default_keypair()
        keypair_path = Path(wallet.KEYPAIR_DIR) / f"{info.name}.json"
        _log_path("Config path", config_path)
        _log_path("Keypair path", keypair_path)
        if info.mnemonic_path:
            _log_path("Mnemonic path", info.mnemonic_path)

    wallet.ensure_default_keypair()
    return config_path, info, keypair_path
