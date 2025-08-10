from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from solhunter_zero.logging_utils import log_startup

__all__ = ["ensure_wallet_cli", "log_startup_info", "run_quick_setup"]

console = Console()


def ensure_wallet_cli() -> None:
    """Ensure the ``solhunter-wallet`` CLI is available."""
    if shutil.which("solhunter-wallet") is not None:
        return
    console.print(
        "[yellow]'solhunter-wallet' command not found. Attempting installation via pip...[/]"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "solhunter-wallet"],
        text=True,
    )
    if result.returncode != 0 or shutil.which("solhunter-wallet") is None:
        console.print(
            "[red]Failed to install 'solhunter-wallet'. Please install it manually with 'pip install solhunter-wallet' and re-run.[/]"
        )
        raise SystemExit(1)


def log_startup_info(*, config_path: Path | None = None, keypair_path: Path | None = None,
                     mnemonic_path: Path | None = None, active_keypair: str | None = None) -> None:
    """Append startup details to ``startup.log``."""
    lines: list[str] = []
    if config_path:
        lines.append(f"Config path: {config_path}")
    if keypair_path:
        lines.append(f"Keypair path: {keypair_path}")
    if mnemonic_path:
        lines.append(f"Mnemonic path: {mnemonic_path}")
    if active_keypair:
        lines.append(f"Active keypair: {active_keypair}")
    if not lines:
        return
    for line in lines:
        log_startup(line)


def run_quick_setup() -> str | None:
    """Run the quick setup non-interactively and return new config path."""
    try:
        from scripts import quick_setup
        from solhunter_zero.config import find_config_file

        quick_setup.main(["--auto", "--non-interactive"])
        return find_config_file()
    except Exception:
        return None
