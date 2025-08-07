from __future__ import annotations

"""Environment preparation helpers for SolHunter Zero.

This module provides a single :func:`prepare` function which performs all
necessary bootstrap steps for running the project.  It wraps the individual
helpers used throughout the codebase so callers have a unified entry point.
"""

import os

from .macos_setup import ensure_tools
from .bootstrap_utils import ensure_venv, ensure_deps
from .build_utils import ensure_route_ffi, ensure_depth_service


def prepare(*, install_optional: bool = False, ensure_wallet_cli: bool = True) -> None:
    """Ensure the runtime environment is fully prepared.

    Parameters
    ----------
    install_optional:
        When ``True`` optional dependencies will be installed in addition to
        the required ones.
    ensure_wallet_cli:
        When ``True`` the ``solhunter-wallet`` command line tool is ensured to
        be available by installing the local package if needed.
    """

    if os.environ.get("SOLHUNTER_SKIP_PREPARE"):
        return

    ensure_tools(non_interactive=True)
    ensure_venv(None)
    ensure_deps(
        install_optional=install_optional, ensure_wallet_cli=ensure_wallet_cli
    )
    ensure_route_ffi()
    ensure_depth_service()
