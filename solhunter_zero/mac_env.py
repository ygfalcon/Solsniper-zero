from __future__ import annotations

from pathlib import Path
from typing import Any

from .logging_utils import log_startup

from scripts import mac_setup

ROOT = Path(__file__).resolve().parent.parent

MANUAL_FIXES = mac_setup.MANUAL_FIXES
apply_brew_env = mac_setup.apply_brew_env
ensure_tools = mac_setup.ensure_tools

__all__ = [
    "prepare_macos_env",
    "apply_brew_env",
    "ensure_tools",
    "MANUAL_FIXES",
]


def prepare_macos_env(*, non_interactive: bool = True) -> dict[str, Any]:
    """Prepare the macOS environment and log each step.

    This wraps :func:`scripts.mac_setup.prepare_macos_env` and ensures the
    Homebrew environment variables are applied consistently.  Each step's
    outcome is recorded in ``startup.log``.
    """

    report = mac_setup.prepare_macos_env(non_interactive=non_interactive)
    mac_setup.apply_brew_env()
    for step, info in report.get("steps", {}).items():
        status = info.get("status")
        message = info.get("message", "")
        if message:
            log_startup(f"mac setup {step}: {status} - {message}")
        else:
            log_startup(f"mac setup {step}: {status}")
    if report.get("success"):
        log_startup("mac setup completed successfully")
    else:
        log_startup("mac setup failed")
    return report
